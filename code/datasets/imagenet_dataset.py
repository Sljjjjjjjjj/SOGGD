import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageNetKeypointDataset(Dataset):
    """
    适用于ImageNet2012数据集的关键点检测数据集类
    """
    def __init__(self, data_root, split='train', img_size=224, augment=True, args=None):
        """
        初始化ImageNet数据集
        
        参数:
            data_root: ImageNet数据集根目录
            split: 'train'或'val'，指定使用训练集或验证集
            img_size: 输入图像大小
            augment: 是否使用数据增强
            args: 其他参数
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.args = args
        
        # 设置数据集路径
        if split == 'train':
            self.data_dir = os.path.join(data_root, 'train')
        else:
            self.data_dir = os.path.join(data_root, 'val')
        
        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                              if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 收集所有图像路径
        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))
        
        # 定义数据变换
        self._build_transforms(augment)
    
    def _build_transforms(self, augment):
        """构建数据变换管道"""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if augment and self.split == 'train':
            self.rgb_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize
            ])
            
            self.grayscale_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize(int(self.img_size * 1.14)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                normalize
            ])
            
            self.grayscale_transform = transforms.Compose([
                transforms.Resize(int(self.img_size * 1.14)),
                transforms.CenterCrop(self.img_size),
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.images[idx]
        
        # 读取图像
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个随机生成的替代图像
            img = Image.fromarray(np.random.randint(0, 256, (self.img_size, self.img_size, 3), dtype=np.uint8))
        
        # 应用变换
        rgb_tensor = self.rgb_transform(img)
        gray_tensor = self.grayscale_transform(img)
        
        # 为关键点检测模型生成热力图
        # 这里使用基于类别和图像特征的伪热力图
        # 实际应用中可以替换为真实的关键点检测热力图
        heatmap = self._generate_pseudo_heatmap(rgb_tensor, gray_tensor, class_idx)
        
        # 组合输入
        input_tensor = torch.cat([rgb_tensor, gray_tensor], dim=0)
        
        return input_tensor, heatmap
    
    def _generate_pseudo_heatmap(self, rgb_tensor, gray_tensor, class_idx):
        """
        生成伪热力图用于训练
        在实际应用中，这应该被替换为真实的关键点热力图生成逻辑
        """
        # 简单的基于梯度的热力图
        h, w = self.img_size, self.img_size
        
        # 使用灰度图作为基础
        base = gray_tensor.squeeze().numpy()
        
        # 应用简单的梯度和高斯模糊
        from scipy.ndimage import gaussian_filter, sobel
        grad_x = sobel(base, axis=1)
        grad_y = sobel(base, axis=0)
        
        # 计算梯度幅度
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 应用高斯模糊
        sigma = self.args.heatmap_sigma if hasattr(self.args, 'heatmap_sigma') else 5.0
        heatmap = gaussian_filter(grad_magnitude, sigma=sigma)
        
        # 归一化到[0,1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # 转换为张量
        heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)
        
        return heatmap_tensor

def imagenet_collate_fn(batch):
    """
    ImageNet数据集的批处理函数
    """
    inputs = torch.stack([item[0] for item in batch])  # [B,4,img_size,img_size]
    targets = torch.stack([item[1] for item in batch])  # [B,1,img_size,img_size]
    return inputs, targets 