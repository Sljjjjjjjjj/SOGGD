import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.keypoint_utils import optimized_blob

class HybridKeypointDataset(Dataset):
    def __init__(self, data_root, img_size=512, augment=True, args=None):
        self.img_size = img_size
        self.augment = augment
        self.args = args

        # Recursively search for all image files
        self.image_paths = []
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')  # Include uppercase extensions
        for ext in valid_exts:
            self.image_paths += glob.glob(os.path.join(data_root, '**', f'*{ext}'), recursive=True)

        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in directory {data_root}!")

        self.transform = self._build_augmentations(augment)

    def _build_augmentations(self, augment):
        spatial_transforms = []
        if augment:
            spatial_transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5)
            ])
        return A.Compose([
            *spatial_transforms,
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(transpose_mask=True)
        ], additional_targets={
            'gray': 'image',
            'heatmap': 'mask'
        })

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        original_h, original_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Scale keypoint coordinates
        keypoints = optimized_blob(gray, o_nb_blobs=self.args.topk)
        if len(keypoints) > 0:
            keypoints[:, 0] = keypoints[:, 0] * (self.img_size / original_h)
            keypoints[:, 1] = keypoints[:, 1] * (self.img_size / original_w)

        heatmap = self.generate_gaussian_heatmap(keypoints)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        transformed = self.transform(
            image=image,
            gray=gray_3ch,
            heatmap=heatmap
        )

        image_tensor = transformed['image']
        gray_tensor = transformed['gray'].mean(dim=0, keepdim=True)
        heatmap_tensor = transformed['heatmap']
        if heatmap_tensor.dim() == 2:
            heatmap_tensor = heatmap_tensor.unsqueeze(0)

        input_tensor = torch.cat([
            transformed['image'],
            transformed['gray'].mean(0, keepdim=True)
        ], dim=0)

        return input_tensor, heatmap_tensor

    def generate_gaussian_heatmap(self, keypoints):
        heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for point in keypoints:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < self.img_size and 0 <= x < self.img_size:
                gaussian = self._create_gaussian(y, x)
                heatmap = np.maximum(heatmap, gaussian)
        return heatmap

    def _create_gaussian(self, cy, cx):
        y = np.arange(0, self.img_size, dtype=np.float32)
        x = np.arange(0, self.img_size, dtype=np.float32)
        y, x = np.meshgrid(y, x, indexing='ij')
        sigma = self.args.heatmap_sigma * (self.img_size / 512)  # Dynamically adjust sigma
        gaussian = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))
        return gaussian / gaussian.max() 