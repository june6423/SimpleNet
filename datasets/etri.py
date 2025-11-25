import os
from enum import Enum
import PIL
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

from .mvtec import MVTecDataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD

_ETRI_ANOMALY_TYPES = [
    "dent",
    "black",
    "scratch",
    "white",
    "stain",
]


class EtriDataset(MVTecDataset):
    """
    Optimized PyTorch Dataset for ETRI Color Plate Data.
    Key optimizations:
    1. GPU-accelerated grid cropping and resizing
    2. Batch-friendly preprocessing
    3. Efficient memory usage
    """

    def __init__(
        self,
        source,
        classname=None,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        grid_patches=3,
        **kwargs,
    ):
        super(MVTecDataset, self).__init__()
        
        self.source = source
        self.split = split
        self.classnames_to_use = ["etri_plate"]
        self.train_val_split = kwargs.get('train_val_split', 1.0)
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        
        self.grid_patches = grid_patches
        self.n_patches_per_image = grid_patches ** 2
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

        # Test Transform - ONLY ToTensor (나머지는 GPU에서 처리)
        self.transform_img_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

        print(f"ETRI Dataset ({self.split.value}) initialized. Found {len(self.data_to_iterate)} base images.")
        if self.split == DatasetSplit.TRAIN:
            print(f"Applying {self.grid_patches}x{self.grid_patches} grid sampling (Train). Total items: {self.__len__()}.")
        else:
            print(f"Applying {self.grid_patches}x{self.grid_patches} grid sampling (Test). Total items: {self.__len__()}.")

    def get_grid_patch(self, image, n_patches, row, col):
        """1024x900 이미지를 3x3 그리드로 자르고 특정 패치를 반환합니다."""
        w, h = image.size
        patch_w = w // n_patches
        patch_h = h // n_patches
        
        left = col * patch_w
        top = row * patch_h
        right = min(left + patch_w, w)
        bottom = min(top + patch_h, h)
        
        patch = image.crop((left, top, right, bottom))
        return patch

    def get_image_data(self):
        data_to_iterate = []
        classname = "etri_plate"

        if self.split == DatasetSplit.TRAIN:
            train_path = os.path.join(self.source, "normal", "normal_train")
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Train path not found: {train_path}")
            
            for file_name in os.listdir(train_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(train_path, file_name)
                    data_to_iterate.append((classname, "good", image_path, None))
        
        else:
            test_path = os.path.join(self.source, "normal", "normal_test")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Train path not found: {test_path}")
            cnt = 0
            for file_name in os.listdir(test_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # cnt += 1
                    # if cnt % 100 != 0:
                    #     continue
                    image_path = os.path.join(test_path, file_name)
                    data_to_iterate.append((classname, "good", image_path, None))

            flaw_root_path = os.path.join(self.source, "flaw", "flaw")
            if not os.path.exists(flaw_root_path):
                print(f"경고: 결함(flaw) 폴더를 찾을 수 없습니다: {flaw_root_path}")
                return {}, []

            for anomaly_type in _ETRI_ANOMALY_TYPES:
                anomaly_path = os.path.join(flaw_root_path, anomaly_type)
                if not os.path.exists(anomaly_path):
                    print(f"  - {anomaly_type} 폴더를 찾을 수 없음, 건너뜁니다.")
                    continue
                
                print(f"  - {anomaly_type} 폴더 로드 중...")
                for file_name in os.listdir(anomaly_path):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # cnt += 1
                        # if cnt % 100 != 0:
                        #     continue
                        image_path = os.path.join(anomaly_path, file_name)
                        data_to_iterate.append((classname, anomaly_type, image_path, None))

        return {}, data_to_iterate

    def grid_crop_gpu(self, img_tensor, n_patches):
        """
        GPU에서 이미지를 그리드로 자르기
        Input: [C, H, W] tensor
        Output: [N_patches, C, patch_H, patch_W] tensor
        """
        C, H, W = img_tensor.shape
        patch_h = H // n_patches
        patch_w = W // n_patches
        
        patches = []
        for row in range(n_patches):
            for col in range(n_patches):
                top = row * patch_h
                left = col * patch_w
                bottom = min(top + patch_h, H)
                right = min(left + patch_w, W)
                
                patch = img_tensor[:, top:bottom, left:right]
                patches.append(patch)
        
        return torch.stack(patches)  # [9, C, patch_H, patch_W]

    def resize_and_crop_gpu(self, img_tensor):
        """
        GPU에서 resize + center crop
        Input: [C, H, W] or [N, C, H, W]
        Output: [C, imagesize, imagesize] or [N, C, imagesize, imagesize]
        """
        # Add batch dimension if needed
        needs_squeeze = False
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
            needs_squeeze = True
        
        # Resize
        resized = F.interpolate(
            img_tensor,
            size=self.resize,
            mode='bilinear',
            align_corners=False
        )
        
        # Center crop
        _, _, h, w = resized.shape
        crop_size = self.imagesize_val
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        cropped = resized[:, :, top:top+crop_size, left:left+crop_size]
        
        if needs_squeeze:
            cropped = cropped.squeeze(0)
        
        return cropped

    def normalize_gpu(self, img_tensor):
        """
        GPU에서 normalization
        Input: [C, H, W] or [N, C, H, W] (0~1 range)
        Output: normalized tensor
        """
        mean = torch.tensor(IMAGENET_MEAN, device=img_tensor.device).view(-1, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=img_tensor.device).view(-1, 1, 1)
        
        if img_tensor.dim() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        
        return (img_tensor - mean) / std

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
            
            patch_row = patch_idx // self.grid_patches
            patch_col = patch_idx % self.grid_patches
            
            classname, anomaly, image_path, mask_path = self.data_to_iterate[file_idx]
            
            image = PIL.Image.open(image_path).convert("RGB")
            patch = self.get_grid_patch(image, self.grid_patches, patch_row, patch_col)
            image = self.transform_img_train(patch)
            
            mask = torch.zeros([1, *image.size()[1:]])
            image_name = f"{Path(image_path).stem}_p{patch_idx}"
            
        else:
            # --- TEST: GPU 가속 처리 ---
            classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
            
            # 1. 이미지 로드 및 ToTensor만 (CPU)
            image_full = PIL.Image.open(image_path).convert("RGB")
            image_tensor = self.transform_img_test(image_full)  # [C, H, W], 0~1
            
            # 2. GPU로 전송 (이 부분은 collate_fn에서 처리하거나, 여기서 미리 처리)
            # NOTE: 여기서 .cuda()를 호출하면 DataLoader의 pin_memory 효과가 사라짐
            # 대신 반환된 텐서를 나중에 GPU로 전송하고, 그 후 처리하는 것이 좋음
            
            # 임시로 여기서는 CPU에서 처리하되, 실제로는 collate_fn에서 GPU 처리 권장
            # 지금은 일단 CPU 버전으로 반환
            image_tensor_cpu = image_tensor
            
            # 마스크 처리
            if mask_path is not None:
                mask_full = PIL.Image.open(mask_path)
                mask = self.transform_mask(mask_full)
            else:
                mask = torch.zeros([1, *self.imagesize[1:]])
            
            image_name = "/".join(image_path.split("/")[-4:])
            
            # Return raw image tensor - processing will be done in custom collate_fn
            return {
                "image": image_tensor_cpu,  # [C, H, W] - 전처리 전
                "mask": mask,
                "classname": classname,
                "anomaly": anomaly,
                "is_anomaly": int(anomaly != "good"),
                "image_name": image_name,
                "image_path": image_path,
            }

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": image_name,
            "image_path": image_path,
        }


# Custom collate function for GPU processing
def etri_collate_fn_gpu(batch, device, grid_patches=3):
    """
    GPU에서 배치 처리를 수행하는 custom collate function
    """
    # Train batch는 그대로 처리
    if "image" in batch[0]:
        return torch.utils.data.dataloader.default_collate(batch)
    
    # Test batch - GPU에서 처리
    batch_size = len(batch)
    
    # 1. 이미지를 GPU로 전송
    images_raw = torch.stack([item["image_raw"] for item in batch]).to(device)  # [B, C, H, W]
    
    # 2. GPU에서 그리드 crop
    all_patches = []
    for i in range(batch_size):
        img = images_raw[i]  # [C, H, W]
        
        # Grid crop
        C, H, W = img.shape
        patch_h = H // grid_patches
        patch_w = W // grid_patches
        
        patches = []
        for row in range(grid_patches):
            for col in range(grid_patches):
                top = row * patch_h
                left = col * patch_w
                bottom = min(top + patch_h, H)
                right = min(left + patch_w, W)
                
                patch = img[:, top:bottom, left:right]
                patches.append(patch)
        
        all_patches.append(torch.stack(patches))  # [9, C, patch_H, patch_W]
    
    images_patches = torch.stack(all_patches)  # [B, 9, C, patch_H, patch_W]
    
    # 3. Resize + Normalize (GPU)
    B, N, C, H, W = images_patches.shape
    images_flat = images_patches.view(B * N, C, H, W)
    
    # Resize to 256
    resized = F.interpolate(images_flat, size=256, mode='bilinear', align_corners=False)
    
    # Center crop to 224
    _, _, h, w = resized.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    cropped = resized[:, :, top:top+224, left:left+224]
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    normalized = (cropped - mean) / std
    
    # Reshape back
    images_final = normalized.view(B, N, C, 224, 224)  # [B, 9, 3, 224, 224]
    
    # 4. 나머지 데이터 collate
    masks = torch.stack([item["mask"] for item in batch])
    
    return {
        "image": images_final,
        "mask": masks,
        "classname": [item["classname"] for item in batch],
        "anomaly": [item["anomaly"] for item in batch],
        "is_anomaly": torch.tensor([item["is_anomaly"] for item in batch]),
        "image_name": [item["image_name"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }