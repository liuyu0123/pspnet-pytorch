import os
import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    # ... 保持原有代码不变 ...
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"{loss}\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(f"{val_loss}\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")


class EvalCallback:
    def __init__(self, model, input_shape, num_classes, val_lines, dataset_path, log_dir, cuda, 
                 eval_flag=True, period=5, image_folder='JPEGImages', 
                 label_folder='SegmentationClass', label_suffix='', label_ext='.png'):
        """
        评估回调，支持自定义标签后缀和扩展名
        """
        self.model = model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.eval_flag = eval_flag
        self.period = period
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.label_suffix = label_suffix
        self.label_ext = label_ext  # 新增
        
        # 构建路径（支持绝对路径）
        if os.path.isabs(image_folder):
            self.images_path = image_folder
        else:
            self.images_path = os.path.join(dataset_path, image_folder)
            
        if os.path.isabs(label_folder):
            self.labels_path = label_folder
        else:
            self.labels_path = os.path.join(dataset_path, label_folder)
        
        self.mious = []
        self.epoches = []
        self.best_miou = 0
        self.best_epoch = 0
        
        # 评估输出目录
        self.miou_out_path = os.path.join(log_dir, "miou_out")
        
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write("epoch\tmiou\n")

    def get_miou_png(self, image):
        # 转换为RGB
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        
        # resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            pr = self.net(images)
            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                pr = pr[1]
            pr = pr[0] if isinstance(pr, (list, tuple)) else pr
            
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            
            # 去除灰条
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
    
        return Image.fromarray(np.uint8(pr))
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            os.makedirs(pred_dir, exist_ok=True)
            
            print("Get miou...")
            for image_id in tqdm(self.val_lines):
                # 构建图片路径（尝试多种扩展名）
                image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    tmp_path = os.path.join(self.images_path, image_id + ext)
                    if os.path.exists(tmp_path):
                        image_path = tmp_path
                        break
                
                if image_path is None:
                    print(f"警告: 找不到图片 {image_id}")
                    continue
                
                # 读取并预测
                image = Image.open(image_path)
                pred = self.get_miou_png(image)
                pred.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou...")
            
            # 构建标签路径列表
            gt_paths = []
            pred_paths = []
            valid_ids = []
            
            for image_id in self.val_lines:
                label_name = image_id + self.label_suffix + self.label_ext
                label_path = os.path.join(self.labels_path, label_name)
                pred_path = os.path.join(pred_dir, image_id + ".png")
                
                if os.path.exists(label_path) and os.path.exists(pred_path):
                    gt_paths.append(label_path)
                    pred_paths.append(pred_path)
                    valid_ids.append(image_id)
            
            if len(valid_ids) == 0:
                print("警告: 没有有效的验证样本")
                return
            
            # 计算 mIoU
            _, IoUs, _, _ = compute_mIoU(self.labels_path, pred_dir, valid_ids, self.num_classes, None)
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            # 保存最佳模型
            if temp_miou > self.best_miou:
                self.best_miou = temp_miou
                self.best_epoch = epoch
                print(f"新的最佳 mIoU: {temp_miou:.2f}% (Epoch {epoch})")

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(f"{epoch}\t{temp_miou:.4f}\n")
            
            # 绘图
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='val miou')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title(f'Best Miou: {self.best_miou:.2f}% (Epoch {self.best_epoch})')
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print(f"Epoch {epoch}: mIoU = {temp_miou:.2f}%")
            
            # 清理临时文件
            shutil.rmtree(self.miou_out_path, ignore_errors=True)