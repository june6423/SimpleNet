#!/bin/bash

echo "Starting GPU-aware parallel execution for 15 MVTec datasets..."
echo "This script will poll GPUs and launch tasks only on free GPUs (97%+ VRAM free)."

# --- 1. 기본 설정 ---

datapath="/workspace/SimpleNet/MVTec_ad"

# 15개의 태스크 (데이터셋 클래스)
# BASH 4.0+ 에서는 readarray를 사용하는 것이 더 좋습니다.
task_queue=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
#task_queue=('pill' 'screw' 'toothbrush')
num_total_tasks=${#task_queue[@]}

available_gpus=(0 1 2 3 4 5 6)
num_gpus=${#available_gpus[@]}

# --- 2. VRAM 확인 설정 ---

VRAM_THRESHOLD=97   # GPU VRAM이 97% *이상* 비어 있어야 함
POLL_INTERVAL=100    # 100초마다 GPU 상태를 확인

# --- 3. 헬퍼 함수 정의 ---

# 함수: 특정 GPU의 비어있는 VRAM 백분율을 반환
get_free_mem_percentage() {
    local gpu_id=$1
    
    # nvidia-smi 쿼리: "memory.free, memory.total" (MiB 단위, 헤더 없음)
    local mem_info=$(nvidia-smi --id=$gpu_id --query-gpu=memory.free,memory.total --format=csv,noheader,nounits)
    
    if [ -z "$mem_info" ]; then
        echo "Error: Could not query GPU $gpu_id. Make sure it exists." >&2
        echo 0
        return
    fi
    
    local free_mem=$(echo $mem_info | awk -F, '{print $1}')
    local total_mem=$(echo $mem_info | awk -F, '{print $2}')
    
    if [ $total_mem -eq 0 ]; then
        echo 0
        return
    fi
    
    # awk를 사용하여 부동 소수점 연산 및 반올림
    local free_pct=$(awk "BEGIN {printf \"%.0f\", ($free_mem / $total_mem) * 100}")
    echo $free_pct
}

# 함수: 실제 파이썬 학습 스크립트를 실행 (백그라운드에서)
launch_task() {
    local gpu_id=$1
    local dataset_name=$2
    local run_name_for_task=$dataset_name

    echo "--- LAUNCHING: $dataset_name on GPU $gpu_id ---"

    # 백그라운드(&)에서 서브셸()로 실행
    (
        python3 main_ours.py \
        --gpu  $gpu_id \
        --seed 0 \
        --log_group simplenet_mvtec \
        --log_project MVTecAD_Results \
        --results_path results_Ours_224_directKD_with_adv \
        --run_name $run_name_for_task \
        net \
        -b wideresnet50 \
        -le layer2 \
        -le layer3 \
        --pretrain_embed_dimension 1536 \
        --target_embed_dimension 1536 \
        --patchsize 3 \
        --meta_epochs 40 \
        --embedding_size 256 \
        --gan_epochs 4 \
        --noise_std 0.015 \
        --dsc_hidden 1024 \
        --dsc_layers 2 \
        --dsc_margin .5 \
        --pre_proj 1 \
        dataset \
        --batch_size 8 \
        --resize 256 \
        --imagesize 224 \
        --augment \
        --brightness 0.1 \
        --contrast 0.1 \
        --saturation 0.1 \
        -d $dataset_name \
        mvtec $datapath
        
        echo "--- FINISHED: $dataset_name on GPU $gpu_id ---"
    ) &
}

# --- 4. 메인 스케줄러 루프 ---

tasks_launched=0
echo "Waiting for tasks to be launched. Total tasks: $num_total_tasks"

# 큐에 작업이 남아있는 동안 계속 실행
while [ ${#task_queue[@]} -gt 0 ]; do
    
    found_free_gpu=false
    
    # 7개의 GPU를 순회하며 빈 GPU를 찾음
    for gpu_id in "${available_gpus[@]}"; do
        
        free_pct=$(get_free_mem_percentage $gpu_id)
        
        if [ $free_pct -ge $VRAM_THRESHOLD ]; then
            # 찾았다! GPU가 97% 이상 비어있음
            found_free_gpu=true
            
            # 큐에서 첫 번째 작업을 가져옴
            dataset_name=${task_queue[0]}
            task_queue=("${task_queue[@]:1}") # BASH에서 배열의 첫 번째 요소 제거 (pop)
            
            # 작업 실행
            launch_task $gpu_id $dataset_name
            tasks_launched=$((tasks_launched + 1))
            
            echo "Launched task $tasks_launched/$num_total_tasks. Remaining tasks: ${#task_queue[@]}"
            
            # 작업을 시작했으므로, VRAM이 할당될 시간을 1초간 주고,
            # 다시 GPU 0번부터 스캔하기 위해 GPU 루프를 탈출
            sleep 1 
            continue
        else
            # 이 GPU는 바쁨
            echo "GPU $gpu_id is busy ($free_pct% free). Checking next GPU."
        fi
    done

    # 만약 모든 GPU를 확인했는데, 빈 GPU가 하나도 없었고,
    # 아직 실행할 작업이 남아있다면...
    if [ "$found_free_gpu" = false ] && [ ${#task_queue[@]} -gt 0 ]; then
        echo "All GPUs are currently busy. Waiting $POLL_INTERVAL seconds to poll again..."
        sleep $POLL_INTERVAL
    fi
    
    # 모든 작업이 큐에서 빠져나가 실행이 시작되면, while 루프가 종료됨.
done

# --- 5. 모든 백그라운드 작업 대기 ---
echo "All $num_total_tasks tasks have been launched. Waiting for all jobs to complete..."
wait
echo "All parallel tasks completed."