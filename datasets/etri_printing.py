import os
from enum import Enum
import PIL
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
import random

from .mvtec import MVTecDataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD

# 모든 폴더 번호 (10~20)
_FOLDER_NUMBERS = list(range(10, 21))  # [10, 11, 12, ..., 20]


class EtriPrintingDataset(MVTecDataset):
    """
    ETRI Printing Plate Dataset.
    - Image resolution: 2690x1000 (very high resolution)
    - Grid: 6x3 = 18 patches per image
    - Train: 11/normal folder only (5000+ images)
    - Test: All flaw folders + 10% of 10/normal
    """

    def __init__(
        self,
        source,
        classname=None,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        grid_patches_x=6,  # 가로 6개
        grid_patches_y=3,  # 세로 3개
        test_normal_ratio=0.1,  # Test에 사용할 normal 데이터 비율
        **kwargs,
    ):
        super(MVTecDataset, self).__init__()
        
        self.source = source
        self.split = split
        self.classnames_to_use = ["etri_printing"]
        self.train_val_split = kwargs.get('train_val_split', 1.0)
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        
        self.grid_patches_x = grid_patches_x
        self.grid_patches_y = grid_patches_y
        self.n_patches_per_image = grid_patches_x * grid_patches_y  # 18
        self.test_normal_ratio = test_normal_ratio
        self.resize = resize
        self.imagesize_val = imagesize
        
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        # Train Transform (Augmentation 포함)
        self.transform_img_train = [
            transforms.Resize(resize),
            transforms.ColorJitter(kwargs.get('brightness_factor', 0), 
                                  kwargs.get('contrast_factor', 0), 
                                  kwargs.get('saturation_factor', 0)),
            transforms.RandomHorizontalFlip(kwargs.get('h_flip_p', 0)),
            transforms.RandomVerticalFlip(kwargs.get('v_flip_p', 0)),
            transforms.RandomGrayscale(kwargs.get('gray_p', 0)),
            transforms.RandomAffine(kwargs.get('rotate_degrees', 0), 
                                    translate=(kwargs.get('translate', 0), kwargs.get('translate', 0)),
                                    scale=(1.0-kwargs.get('scale', 0), 1.0+kwargs.get('scale', 0)),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img_train = transforms.Compose(self.transform_img_train)

        # Test Transform - ONLY ToTensor (나머지는 CPU에서 처리)
        self.transform_img_test = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img_test = transforms.Compose(self.transform_img_test)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

        print(f"ETRI Printing Dataset ({self.split.value}) initialized. Found {len(self.data_to_iterate)} base images.")
        if self.split == DatasetSplit.TRAIN:
            print(f"Applying {self.grid_patches_x}x{self.grid_patches_y} grid sampling (Train). Total items: {self.__len__()}.")
        else:
            print(f"Applying {self.grid_patches_x}x{self.grid_patches_y} grid sampling (Test). Total items: {self.__len__()}.")

    def get_grid_patch(self, image, patches_x, patches_y, row, col):
        """
        2690x1000 이미지를 6x3 그리드로 자르고 특정 패치를 반환합니다.
        Args:
            image: PIL Image
            patches_x: 가로 패치 개수 (6)
            patches_y: 세로 패치 개수 (3)
            row: 패치 행 인덱스 (0~2)
            col: 패치 열 인덱스 (0~5)
        """
        w, h = image.size
        patch_w = w // patches_x
        patch_h = h // patches_y
        
        left = col * patch_w
        top = row * patch_h
        right = min(left + patch_w, w)
        bottom = min(top + patch_h, h)
        
        patch = image.crop((left, top, right, bottom))
        return patch

    def get_image_data(self):
        data_to_iterate = []
        classname = "etri_printing"

        if self.split == DatasetSplit.TRAIN:
            # Train: 11/normal 폴더만 사용
            train_path = os.path.join(self.source, "11", "normal")
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Train path not found: {train_path}")
            
            count = 0
            for file_name in sorted(os.listdir(train_path)):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(train_path, file_name)
                    data_to_iterate.append((classname, "good", image_path, None))
                    count += 1
            
            print(f"Loaded {count} training images from 11/normal")
        
        else:
            # Test: 10/normal의 10% + 모든 flaw 폴더
            
            # 1. 10/normal의 10% 샘플링
            test_normal_path = os.path.join(self.source, "10", "normal")
            if os.path.exists(test_normal_path):
                normal_files = sorted([
                    f for f in os.listdir(test_normal_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ])
                
                # 10% 샘플링 (랜덤하지만 재현 가능하도록 seed 고정)
                random.seed(42)
                num_samples = max(1, int(len(normal_files) * self.test_normal_ratio))
                sampled_files = random.sample(normal_files, num_samples)
                
                for file_name in sampled_files:
                    image_path = os.path.join(test_normal_path, file_name)
                    data_to_iterate.append((classname, "good", image_path, None))
                
                print(f"Loaded {len(sampled_files)} normal test images from 10/normal ({self.test_normal_ratio*100:.0f}%)")
            else:
                print(f"Warning: 10/normal path not found: {test_normal_path}")
            
            # 2. 모든 폴더의 flaw 데이터 로드 (단일 클래스로 처리)
            total_flaw_count = 0
            
            for folder_num in _FOLDER_NUMBERS:
                flaw_path = os.path.join(self.source, str(folder_num), "flaw")
                
                if not os.path.exists(flaw_path):
                    continue
                
                folder_flaw_count = 0
                
                # flaw 폴더 내부의 모든 이미지를 재귀적으로 로드
                for root, dirs, files in os.walk(flaw_path):
                    for file_name in files:
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            image_path = os.path.join(root, file_name)
                            # 모든 flaw를 "flaw"로 통일
                            data_to_iterate.append((classname, "flaw", image_path, None))
                            folder_flaw_count += 1
                            total_flaw_count += 1
                
                if folder_flaw_count > 0:
                    print(f"  - Folder {folder_num}/flaw: {folder_flaw_count} images")
            
            print(f"Total flaw images: {total_flaw_count}")

        return {}, data_to_iterate

    def __len__(self):
        if self.split == DatasetSplit.TRAIN:
            return len(self.data_to_iterate) * self.n_patches_per_image
        else:
            return len(self.data_to_iterate)
            
    def __getitem__(self, idx):
        if self.split == DatasetSplit.TRAIN:
            # --- TRAIN: Grid Sampling (개별 패치) ---
            file_idx = idx // self.n_patches_per_image
            patch_idx = idx % self.n_patches_per_image
            
            patch_row = patch_idx // self.grid_patches_x  # 0~2
            patch_col = patch_idx % self.grid_patches_x   # 0~5
            
            classname, anomaly, image_path, mask_path = self.data_to_iterate[file_idx]
            
            image = PIL.Image.open(image_path).convert("RGB")  # 2690x1000
            patch = self.get_grid_patch(image, self.grid_patches_x, self.grid_patches_y, patch_row, patch_col)
            image = self.transform_img_train(patch)
            
            mask = torch.zeros([1, *image.size()[1:]])
            image_name = f"{Path(image_path).stem}_p{patch_idx}"
            
        else:
            # --- TEST: 18개 패치 스택 ---
            classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
            image_full = PIL.Image.open(image_path).convert("RGB")  # 2690x1000
            
            patch_list = []
            for i in range(self.n_patches_per_image):  # 0..17
                patch_row = i // self.grid_patches_x  # 0~2
                patch_col = i % self.grid_patches_x   # 0~5
                
                patch = self.get_grid_patch(image_full, self.grid_patches_x, self.grid_patches_y, patch_row, patch_col)
                patch_tensor = self.transform_img_test(patch)
                patch_list.append(patch_tensor)
            
            # [18, 3, 224, 224] 텐서 스택
            image = torch.stack(patch_list)

            if mask_path is not None:
                mask_full = PIL.Image.open(mask_path)
                mask = self.transform_mask(mask_full)
            else:
                mask = torch.zeros([1, *self.imagesize[1:]])
            
            image_name = "/".join(image_path.split("/")[-4:])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": image_name,
            "image_path": image_path,
        }