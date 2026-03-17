import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input

class PSPnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, 
                 dataset_path=None, image_folder='JPEGImages', label_folder='SegmentationClass', 
                 label_suffix='', label_ext='.png', is_2007=False):
        """
        修改后的数据集类，支持：
        1. 传统 VOC 格式（is_2007=True）
        2. 自定义路径格式（传入完整路径作为 image_folder/label_folder）
        3. 相对路径格式（dataset_path + image_folder/label_folder）
        """
        super(PSPnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.label_suffix       = label_suffix
        self.label_ext          = label_ext
        
        # 构建完整路径（优先级：绝对路径 > VOC2007格式 > 普通相对路径）
        
        # 图片路径处理
        if os.path.isabs(image_folder):
            # 传入的是完整绝对路径
            self.images_path = image_folder
        elif is_2007 and dataset_path is not None:
            # VOC2007 格式: dataset_path/VOC2007/image_folder
            self.images_path = os.path.join(dataset_path, "VOC2007", image_folder)
        elif dataset_path is not None:
            # 普通相对路径: dataset_path/image_folder
            self.images_path = os.path.join(dataset_path, image_folder)
        else:
            raise ValueError("无法确定图片路径。请传入绝对路径的 image_folder，或提供 dataset_path")
        
        # 标签路径处理
        if os.path.isabs(label_folder):
            # 传入的是完整绝对路径
            self.labels_path = label_folder
        elif is_2007 and dataset_path is not None:
            self.labels_path = os.path.join(dataset_path, "VOC2007", label_folder)
        elif dataset_path is not None:
            self.labels_path = os.path.join(dataset_path, label_folder)
        else:
            raise ValueError("无法确定标签路径。请传入绝对路径的 label_folder，或提供 dataset_path")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        name = self.annotation_lines[index].strip()
        
        # 构建图片路径（尝试多种扩展名）
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            tmp_path = os.path.join(self.images_path, name + ext)
            if os.path.exists(tmp_path):
                image_path = tmp_path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"找不到图片文件: {name}.* (在 {self.images_path} 中)")
        
        # 构建标签路径：name + suffix + label_ext
        label_name = name + self.label_suffix + self.label_ext
        label_path = os.path.join(self.labels_path, label_name)
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"找不到标签文件: {label_name} (在 {self.labels_path} 中)")

        #-------------------------------#
        #   读取图片和标签
        #-------------------------------#
        # 读取图片
        jpg = Image.open(image_path)
        
        # 读取标签（支持 gif）
        png = Image.open(label_path)
        
        # gif 格式需要特殊处理（取第一帧，转为灰度）
        if label_path.lower().endswith('.gif'):
            png.seek(0)  # 取第一帧
            png = png.convert('L')  # 转为灰度图

        jpg = cvtColor(jpg)
        # 确保 label 是单通道灰度图
        if png.mode != 'L':
            png = png.convert('L')
        
        # 数据增强
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        #-------------------------------#
        #   转换为 Tensor
        #-------------------------------#
        
        # 1. 处理图片 (jpg)
        jpg_np = preprocess_input(np.array(jpg, np.float64))
        jpg_tensor = torch.from_numpy(np.transpose(jpg_np, [2, 0, 1])).float()

        # 2. 处理标签 (png) - 类别索引
        png_np = np.array(png)
        # 确保标签值在合法范围内 [0, num_classes]
        png_np[png_np >= self.num_classes] = self.num_classes - 1  # 修正：应该是 num_classes-1
        png_tensor = torch.from_numpy(png_np).long()

        # 3. 处理 seg_labels (One-hot 编码)
        seg_labels_np = np.eye(self.num_classes + 1)[png_np.reshape([-1])]
        seg_labels_np = seg_labels_np.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        seg_labels_tensor = torch.from_numpy(seg_labels_np).float()

        return jpg_tensor, png_tensor, seg_labels_tensor

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        image = image.convert('RGB')
        label = label.convert('L')
        
        iw, ih = image.size
        h, w = input_shape

        if not random:
            # 非训练模式：直接 resize
            image = image.resize((w, h), Image.BICUBIC)
            label = label.resize((w, h), Image.NEAREST)
            return image, label

        # 数据增强（训练模式）
        # 随机缩放
        new_ar = w/h * (1 + np.random.uniform(-jitter, jitter))
        scale = np.random.uniform(0.5, 2)
        
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
            
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        
        # 随机裁剪或填充
        dx = int(np.random.uniform(0, nw - w)) if nw > w else int(np.random.uniform(nw - w, 0))
        dy = int(np.random.uniform(0, nh - h)) if nh > h else int(np.random.uniform(nh - h, 0))
        
        # 创建新图像
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), 0)
        
        # 粘贴
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        
        image = new_image
        label = new_label
        
        return image, label


def pspnet_dataset_collate(batch):
    """
    Collate 函数直接堆叠为 Tensor
    """
    images = []
    pngs = []
    seg_labels = []
    
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    
    images = torch.stack(images, 0)
    pngs = torch.stack(pngs, 0)
    seg_labels = torch.stack(seg_labels, 0)
    
    return images, pngs, seg_labels