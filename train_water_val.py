import datetime
import os
import random
import argparse
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.pspnet import PSPNet
from nets.pspnet_training import (get_lr_scheduler, set_optimizer_lr,
                                  weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import PSPnetDataset, pspnet_dataset_collate
from utils.utils import (download_weights, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch


def get_files_from_dir(image_dir, label_dir, label_ext='.png', label_suffix=''):
    """
    从目录中读取所有有效的图片-标签对
    """
    valid_files = []
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    if not os.path.exists(image_dir):
        raise ValueError(f"图片目录不存在: {image_dir}")
    if not os.path.exists(label_dir):
        raise ValueError(f"标签目录不存在: {label_dir}")
    
    for fname in os.listdir(image_dir):
        is_image = any(fname.lower().endswith(ext) for ext in image_exts)
        if not is_image:
            continue
            
        name_without_ext = os.path.splitext(fname)[0]
        label_fname = name_without_ext + label_suffix + label_ext
        label_path = os.path.join(label_dir, label_fname)
        
        if os.path.exists(label_path):
            valid_files.append(name_without_ext)
        else:
            print(f"警告: 找不到标签文件 {label_path}，跳过 {fname}")
    
    if len(valid_files) == 0:
        raise ValueError(f"在 {image_dir} 中没有找到有效的图片-标签对！")
    
    valid_files.sort()
    return valid_files


def split_dataset(image_dir, val_split=0.1, seed=42, label_dir=None, 
                  image_exts=['.jpg', '.jpeg', '.png', '.bmp'], 
                  label_ext='.png', label_suffix=''): 
    """
    自动分割数据集（从同一目录划分）
    """
    random.seed(seed)
    
    if label_dir is None:
        label_dir = image_dir
    
    valid_files = get_files_from_dir(image_dir, label_dir, label_ext, label_suffix)
    
    random.shuffle(valid_files)
    val_num = int(len(valid_files) * val_split)
    val_lines = valid_files[:val_num]
    train_lines = valid_files[val_num:]
    
    print(f"\n{'='*50}")
    print(f"数据集自动分割完成:")
    print(f"  总样本数: {len(valid_files)}")
    print(f"  训练集: {len(train_lines)} ({len(train_lines)/len(valid_files)*100:.1f}%)")
    print(f"  验证集: {len(val_lines)} ({len(val_lines)/len(valid_files)*100:.1f}%)")
    print(f"{'='*50}\n")
    
    return train_lines, val_lines


def parse_args():
    parser = argparse.ArgumentParser(description='PSPNet 水域分割训练')
    
    # 数据路径参数（新的命令行方式）
    parser.add_argument('--images', type=str, default=None,
                        help='训练集图片目录路径')
    parser.add_argument('--masks', type=str, default=None,
                        help='训练集标签目录路径')
    parser.add_argument('--val-images', type=str, default=None,
                        help='验证集图片目录路径（可选，默认从训练集划分）')
    parser.add_argument('--val-masks', type=str, default=None,
                        help='验证集标签目录路径（可选）')
    
    # 兼容旧版的数据集根目录配置
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='数据集根目录（旧版配置方式）')
    parser.add_argument('--image-folder', type=str, default='images',
                        help='图片文件夹名（相对于根目录）')
    parser.add_argument('--mask-folder', type=str, default='masks',
                        help='标签文件夹名（相对于根目录）')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='验证集比例（当未指定--val-images时使用）')
    
    # 标签格式
    parser.add_argument('--mask-ext', type=str, default='.png',
                        help='标签文件扩展名')
    parser.add_argument('--mask-suffix', type=str, default='',
                        help='标签文件后缀（如 _mask）')
    
    # 训练基本配置
    parser.add_argument('--epochs', type=int, default=5,
                        help='总训练轮数')
    parser.add_argument('--freeze-epochs', type=int, default=5,
                        help='冻结阶段轮数')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='解冻阶段batch size')
    parser.add_argument('--freeze-batch-size', type=int, default=8,
                        help='冻结阶段batch size')
    parser.add_argument('--input-height', type=int, default=320,
                        help='输入图片高度')
    parser.add_argument('--input-width', type=int, default=640,
                        help='输入图片宽度')
    
    # 模型配置
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['mobilenet', 'resnet50'],
                        help='主干网络')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='分割类别数')
    parser.add_argument('--pretrained', action='store_true',
                        help='使用预训练权重')
    parser.add_argument('--model-path', type=str, default='',
                        help='加载已有模型权重')
    
    # 优化器配置
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='初始学习率')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='优化器类型')
    
    # 其他
    parser.add_argument('--save-dir', type=str, default='logs',
                        help='日志保存目录')
    parser.add_argument('--seed', type=int, default=11,
                        help='随机种子')
    parser.add_argument('--no-freeze', action='store_true',
                        help='不进行冻结训练')
    parser.add_argument('--fp16', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载线程数')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #---------------------------------#
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = args.fp16
    
    # 从命令行参数获取配置
    num_classes = args.num_classes
    backbone = args.backbone
    pretrained = args.pretrained
    model_path = args.model_path
    downsample_factor = 16
    input_shape = [args.input_height, args.input_width]
    
    # 训练参数
    Init_Epoch = 0
    Freeze_Epoch = args.freeze_epochs if not args.no_freeze else 0
    Freeze_batch_size = args.freeze_batch_size
    UnFreeze_Epoch = args.epochs
    Unfreeze_batch_size = args.batch_size
    Freeze_Train = not args.no_freeze
    
    # 优化器参数
    Init_lr = args.lr
    Min_lr = Init_lr * 0.01
    optimizer_type = args.optimizer
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay_type = 'cos'
    save_period = 5
    save_dir = args.save_dir
    eval_flag = True
    eval_period = 5
    
    # 标签格式
    label_ext = args.mask_ext
    label_suffix = args.mask_suffix
    num_workers = args.workers
    
    seed_everything(args.seed)
    
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    if local_rank == 0:
        print(f"使用设备: {device}")
        print(f"输入尺寸: {input_shape}")

    # 下载预训练权重
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model = PSPNet(num_classes=num_classes, backbone=backbone, 
                   downsample_factor=downsample_factor, 
                   pretrained=pretrained, aux_branch=False)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # 加载权重代码...
        print(f"加载权重: {model_path}")
        # ... 原有加载逻辑
        pass

    # 记录Loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, 
                                                                    device_ids=[local_rank], 
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   数据集路径处理（核心修改）
    #---------------------------#
    
    # 优先级1: 使用命令行指定的独立 train/val 路径
    if args.images is not None and args.masks is not None:
        train_image_dir = args.images
        train_label_dir = args.masks
        
        # 如果指定了独立的 val 路径
        if args.val_images is not None and args.val_masks is not None:
            val_image_dir = args.val_images
            val_label_dir = args.val_masks
            
            if local_rank == 0:
                print(f"\n{'='*50}")
                print(f"使用独立 Train/Val 路径模式")
                print(f"训练图片: {train_image_dir}")
                print(f"训练标签: {train_label_dir}")
                print(f"验证图片: {val_image_dir}")
                print(f"验证标签: {val_label_dir}")
                print(f"{'='*50}")
            
            train_lines = get_files_from_dir(train_image_dir, train_label_dir, 
                                             label_ext, label_suffix)
            val_lines = get_files_from_dir(val_image_dir, val_label_dir, 
                                           label_ext, label_suffix)
        
        # 只指定了 train，需要自动划分
        else:
            if local_rank == 0:
                print(f"\n{'='*50}")
                print(f"从训练集自动划分验证集 (比例: {args.val_split})")
                print(f"图片目录: {train_image_dir}")
                print(f"标签目录: {train_label_dir}")
                print(f"{'='*50}")
            
            train_lines, val_lines = split_dataset(
                train_image_dir,
                val_split=args.val_split,
                seed=args.seed,
                label_dir=train_label_dir,
                label_ext=label_ext,
                label_suffix=label_suffix
            )
            # 验证集使用相同目录
            val_image_dir = train_image_dir
            val_label_dir = train_label_dir
    
    # 优先级2: 使用旧版根目录配置
    elif args.dataset_root is not None:
        dataset_root = args.dataset_root
        image_folder = args.image_folder
        mask_folder = args.mask_folder
        
        train_image_dir = os.path.join(dataset_root, image_folder)
        train_label_dir = os.path.join(dataset_root, mask_folder)
        val_image_dir = train_image_dir
        val_label_dir = train_label_dir
        
        if local_rank == 0:
            print(f"\n{'='*50}")
            print(f"使用数据集根目录模式")
            print(f"根目录: {dataset_root}")
            print(f"自动划分验证集 (比例: {args.val_split})")
            print(f"{'='*50}")
        
        train_lines, val_lines = split_dataset(
            train_image_dir,
            val_split=args.val_split,
            seed=args.seed,
            label_dir=train_label_dir,
            label_ext=label_ext,
            label_suffix=label_suffix
        )
    
    else:
        raise ValueError("必须指定数据路径！请使用以下方式之一：\n"
                        "1. --images 和 --masks（推荐）\n"
                        "2. --dataset-root（旧版方式）")

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        print(f"\n最终数据集:")
        print(f"  训练样本: {num_train}")
        print(f"  验证样本: {num_val}")
        
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, 
            input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, 
            UnFreeze_Epoch=UnFreeze_Epoch, 
            Freeze_batch_size=Freeze_batch_size, 
            Unfreeze_batch_size=Unfreeze_batch_size, 
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, 
            optimizer_type=optimizer_type, momentum=momentum, 
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, 
            num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (total_step, wanted_step, wanted_epoch))
        
    #------------------------------------------------------#
    #   冻结训练设置
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, 
                              betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, 
                            momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # 创建数据集 - 修改后的 PSPnetDataset 调用
        train_dataset = PSPnetDataset(
            train_lines, input_shape, num_classes, True, 
            train_image_dir,  # 直接传入完整路径
            '',  # image_folder 设为空（已包含在路径中）
            train_label_dir,  # 直接传入完整路径
            label_suffix, 
            label_ext, 
            is_2007=False
        )
        val_dataset = PSPnetDataset(
            val_lines, input_shape, num_classes, False, 
            val_image_dir,
            '',
            val_label_dir,
            label_suffix, 
            label_ext, 
            is_2007=False
        )
        
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, 
                        num_workers=num_workers, pin_memory=True,
                        drop_last=True, collate_fn=pspnet_dataset_collate, 
                        sampler=train_sampler, 
                        worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, 
                            num_workers=num_workers, pin_memory=True, 
                            drop_last=True, collate_fn=pspnet_dataset_collate, 
                            sampler=val_sampler, 
                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))

        if local_rank == 0:
            eval_callback = EvalCallback(
                model, input_shape, num_classes, val_lines, 
                val_image_dir,  # 传入验证集图片目录
                log_dir, Cuda,
                eval_flag=eval_flag, period=eval_period, 
                image_folder='',  # 设为空
                label_folder='',  # 设为空
                label_suffix=label_suffix, 
                label_ext=label_ext
            )
        else:
            eval_callback = None
        
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 16
                lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, 
                                num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=pspnet_dataset_collate, 
                                sampler=train_sampler, 
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, 
                                    num_workers=num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=pspnet_dataset_collate, 
                                    sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, 
                    False, False, np.ones([num_classes], np.float32), False, 
                    num_classes, fp16, scaler, save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()