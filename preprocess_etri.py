import os
import PIL.Image
from torchvision import transforms
from tqdm import tqdm
import glob
import torch


# --- 설정 ---
SOURCE_ROOT = "/home/vision/dongjun/SimpleNet/colorplate"
TARGET_ROOT = "/home/vision/dongjun/SimpleNet/colorplate_preprocessed_test" # 새 경로
ORIG_W, ORIG_H = 1024, 900
PATCH_GRID_SIZE = 3
TARGET_IMG_SIZE = 224
RESIZE = 256

# 전처리에 사용할 변환 (ToTensor + Normalize 제외)
transform_preprocess = transforms.Compose([
    transforms.Resize(RESIZE),
    transforms.CenterCrop(TARGET_IMG_SIZE),
])

# ToTensor + Normalize (PT 파일 저장 *후에* Dataloader가 적용)
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# --- ---

def _get_patch_bbox(patch_idx, grid_size, orig_w, orig_h):
    patch_w = orig_w // grid_size
    patch_h = orig_h // grid_size
    
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    
    start_x = col * patch_w
    start_y = row * patch_h
    end_x = (col + 1) * patch_w
    end_y = (row + 1) * patch_h
    
    if col == grid_size - 1:
        end_x = orig_w
    if row == grid_size - 1:
        end_y = orig_h

    return (start_x, start_y, end_x, end_y)

def process_folder_to_pt(source_dir, target_dir):
    """
    [MODIFIED]
    source_dir의 이미지를 9개 패치로 잘라 *하나의 .pt 파일*로 저장합니다.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"Processing {source_dir} -> {target_dir}")
    
    image_paths = glob.glob(os.path.join(source_dir, "*.jpg"))
    if not image_paths:
        print(f"Warning: No images found in {source_dir}")
        return

    for image_path in tqdm(image_paths, desc=f"Folder {os.path.basename(source_dir)}"):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        try:
            image = PIL.Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}, skipping.")
            continue

        patch_pil_list = [] # PIL 이미지 리스트
        for i in range(PATCH_GRID_SIZE * PATCH_GRID_SIZE): # 0~8
            patch_bbox = _get_patch_bbox(i, PATCH_GRID_SIZE, image.width, image.height)
            patch = image.crop(patch_bbox)
            
            # [MODIFIED] ToTensor/Normalize는 Dataloader가 하도록 PIL 이미지 상태로 둠
            patch_transformed = transform_preprocess(patch) # 224x224 PIL 이미지
            patch_pil_list.append(patch_transformed)
        
        # [MODIFIED] 9개의 PIL 이미지를 .pt 파일 하나로 저장
        # Dataloader가 ToTensor/Normalize를 적용할 수 있도록 PIL 이미지 리스트를 저장
        target_filename = f"{base_name}.pt"
        target_path = os.path.join(target_dir, target_filename)
        
        torch.save(patch_pil_list, target_path)
            
# --- 스크립트 실행 ---
# print("Starting TEST SET preprocessing...")

# # 1. Normal (Test) 처리
src_normal_test = os.path.join(SOURCE_ROOT, "normal", "normal_test")
tgt_normal_test = os.path.join(TARGET_ROOT, "normal_test_patches")
process_folder_to_pt(src_normal_test, tgt_normal_test)

# 2. Anomalies (Flaw) 처리
src_flaw_root = os.path.join(SOURCE_ROOT, "flaw", "flaw")
tgt_flaw_root = os.path.join(TARGET_ROOT, "flaw_patches")

if os.path.exists(src_flaw_root):
    anomaly_types = [d for d in os.listdir(src_flaw_root) if os.path.isdir(os.path.join(src_flaw_root, d))]
    for anomaly_type in anomaly_types:
        src_anomaly_dir = os.path.join(src_flaw_root, anomaly_type)
        tgt_anomaly_dir = os.path.join(tgt_flaw_root, anomaly_type)
        process_folder_to_pt(src_anomaly_dir, tgt_anomaly_dir)
else:
    print(f"Warning: Flaw directory not found at {src_flaw_root}")

print("Test set preprocessing complete.")

# print("Starting TRAIN SET preprocessing...")

# # 1. Normal (Test) 처리
# src_normal_test = os.path.join(SOURCE_ROOT, "normal", "normal_train")
# tgt_normal_test = os.path.join(TARGET_ROOT, "normal_train_patches")
# process_folder(src_normal_test, tgt_normal_test)


# print("Test set preprocessing complete.")