import os
from enum import Enum
import glob
import PIL.Image
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path

# MVTecDataset 클래스를 더 이상 상속받지 않습니다.
# from .mvtec import MVTecDataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD

# [NEW] 필요한 변수들을 이 파일에 직접 정의합니다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

_ETRI_ANOMALY_TYPES = [
    "dent", "white", "stain", "black", "scratch",
]

class EtriDataset(Dataset):
    """
    [MODIFIED for ETRI Color Plate - Static Loading OOM Fix]
    - TRAIN: Loads pre-processed 224x224 patches from `normal_train_patches`.
    - TEST:  Loads pre-processed 9-patch groups from `colorplate_preprocessed_test`.
    - This class eliminates dynamic loading of 1024x900 images,
      fixing both 'Killed' (RAM OOM) and '28-hour' (CPU bottleneck) issues.
    """

    def __init__(
        self,
        source, # e.g., ".../colorplate"
        classname=None, # (무시됨)
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        grid_patches=3,
        **kwargs,
    ):
        super(EtriDataset, self).__init__() 
        
        self.source = source
        self.split = split
        self.grid_patches = grid_patches
        self.n_patches_per_image = grid_patches ** 2
        
        # [NEW] 전처리된 테스트 폴더 경로
        # (원본 'colorplate' 폴더의 부모 디렉터리에 있다고 가정)
        self.source = source

        # --- 1. Train/Test용 변환 정의 ---
        
        # 1. Train Transform (Augmentation)
        # (이미 224x224이므로 Resize/Crop 없음)
        self.transform_img_train = transforms.Compose([
            transforms.ColorJitter(kwargs.get('brightness_factor', 0), 
                                  kwargs.get('contrast_factor', 0), 
                                  kwargs.get('saturation_factor', 0)),
            transforms.RandomAffine(kwargs.get('rotate_degrees', 0), 
                                    translate=(kwargs.get('translate', 0), kwargs.get('translate', 0)),
                                    scale=(1.0-kwargs.get('scale', 0), 1.0+kwargs.get('scale', 0)),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(kwargs.get('h_flip_p', 0)),
            transforms.RandomVerticalFlip(kwargs.get('v_flip_p', 0)),
            transforms.RandomGrayscale(kwargs.get('gray_p', 0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # 2. Test Transform (No Augmentation)
        # (이미 224x224이므로 Resize/Crop 없음)
        self.transform_img_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize(resize), # 마스크는 여전히 원본 기준이므로 리사이즈 필요
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])
        
        self.imagesize = (3, imagesize, imagesize)
        
        # --- 2. 데이터 스캔 ---
        self.data_to_iterate = self.get_image_data()
        
        print(f"ETRI Dataset ({self.split.value}) initialized.")
        if self.split == DatasetSplit.TRAIN:
            print(f"Found {len(self.data_to_iterate)} pre-processed TRAIN patches.")
        else:
            print(f"Found {len(self.data_to_iterate)} base TEST images (total {len(self.data_to_iterate) * self.n_patches_per_image} patches).")

    def get_image_data(self):
        """
        [MODIFIED]
        - TRAIN: `normal_train_patches`에서 개별 패치 경로를 스캔합니다.
        - TEST: `colorplate_preprocessed_test`에서 9개 패치 *그룹*을 스캔합니다.
        """
        data_to_iterate = []
        classname = "etri_plate" # (고정)

        if self.split == DatasetSplit.TRAIN:
            # --- 1. Train 데이터 로드 (전처리된 패치) ---
            img_dir = os.path.join(self.source, "normal_train_patches")
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Directory not found: {img_dir}. 'preprocess_etri_patches.py'를 먼저 실행하세요.")
            
            # 모든 .jpg 파일을 스캔
            image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
            for img_path in image_paths:
                # (classname, anomaly, image_path, mask_path)
                data_to_iterate.append((classname, "good", img_path, None))
        
        else: 
            # --- 2. Test 데이터 로드 (전처리된 패치 *경로* 스캔) ---
            if not os.path.exists(self.source):
                raise FileNotFoundError(f"Test preprocessed dir not found: {self.source}. 'preprocess_etri_test.py'를 먼저 실행하세요.")
            
            # 2a. Normal (Test)
            normal_test_dir = os.path.join(self.source, "normal_test_patches")
            # `img_001_p0.jpg` ... `img_001_p8.jpg`에서 `img_001`을 추출
            normal_base_names = sorted(list(set([
                "_".join(f.split('_')[:-1]) for f in os.listdir(normal_test_dir) if f.endswith('.jpg')
            ])))
            for base_name in normal_base_names:
                # 9개 패치의 경로 리스트를 생성
                patch_paths = [os.path.join(normal_test_dir, f"{base_name}_p{i}.jpg") for i in range(self.n_patches_per_image)]
                # (classname, anomaly, list_of_9_paths, mask_path)
                data_to_iterate.append((classname, "good", patch_paths, None))
            
            # 2b. Anomalies (Flaw)
            flaw_root_path = os.path.join(self.source, "flaw_patches")
            for anomaly_type in _ETRI_ANOMALY_TYPES:
                anomaly_path = os.path.join(flaw_root_path, anomaly_type)
                if not os.path.exists(anomaly_path):
                    print(f"Warning: Anomaly dir not found: {anomaly_path}, skipping.")
                    continue
                
                # `defect_001_p0.jpg` ... `defect_001_p8.jpg`에서 `defect_001`을 추출
                anomaly_base_names = sorted(list(set([
                    "_".join(f.split('_')[:-1]) for f in os.listdir(anomaly_path) if f.endswith('.jpg')
                ])))
                for base_name in anomaly_base_names:
                    patch_paths = [os.path.join(anomaly_path, f"{base_name}_p{i}.jpg") for i in range(self.n_patches_per_image)]
                    data_to_iterate.append((classname, anomaly_type, patch_paths, None))
        
        return data_to_iterate # (imgpaths_per_class는 사용하지 않음)

    def __len__(self):
        # TRAIN, TEST 모두 `data_to_iterate`의 길이를 반환
        return len(self.data_to_iterate)
            
    def __getitem__(self, idx):
        classname, anomaly, image_path_or_list, mask_path = self.data_to_iterate[idx]
        is_anomaly = int(anomaly != "good")

        try:
            if self.split == DatasetSplit.TRAIN:
                # --- TRAIN: 단일 패치 로드 (빠름, RAM 안전) ---
                image_path = image_path_or_list
                image = PIL.Image.open(image_path).convert("RGB")
                image = self.transform_img_train(image) # Augmentation 적용
                mask = torch.zeros([1, *image.size()[1:]])
                image_name = Path(image_path).stem
                
            else:
                # --- TEST: 9개 패치 로드 및 스택 (빠름, RAM 안전) ---
                patch_list = []
                image_path_list = image_path_or_list # 9개 경로 리스트
                
                for patch_path in image_path_list:
                    patch = PIL.Image.open(patch_path).convert("RGB")
                    patch_tensor = self.transform_img_test(patch) # No Augmentation
                    patch_list.append(patch_tensor)
                
                image = torch.stack(patch_list) # [9, 3, 224, 224]
                
                # (마스크 로직은 ETRI에 없으므로 0으로 채움)
                mask = torch.zeros([1, *self.imagesize[1:]]) 
                image_name = Path(image_path_list[0]).stem.replace("_p0", "")
                
        except Exception as e:
            print(f"Error loading image data for idx {idx} (path: {image_path_or_list}): {e}")
            # DataLoader가 붕괴하지 않도록 더미 텐서 반환
            if self.split == DatasetSplit.TRAIN:
                return {
                    "image": torch.zeros(3, *self.imagesize[1:]),
                    "mask": torch.zeros(1, *self.imagesize[1:]),
                    "classname": "error", "anomaly": "error", "is_anomaly": 1,
                    "image_name": "error", "image_path": "error"
                }
            else:
                return {
                    "image": torch.zeros(self.n_patches_per_image, 3, *self.imagesize[1:]),
                    "mask": torch.zeros(1, *self.imagesize[1:]),
                    "classname": "error", "anomaly": "error", "is_anomaly": 1,
                    "image_name": "error", "image_path": "error"
                }

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": is_anomaly,
            "image_name": image_name,
            "image_path": str(image_path_or_list), # (Test는 리스트의 str 표현)
        }