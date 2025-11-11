datapath=/workspace/SimpleNet/MVTec_ad
datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
#datasets=('screw' )
#datasets=( 'capsule' 'grid' 'tile' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' )

dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main_ours.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_mvtec \
--log_project MVTecAD_Results \
--results_path results_Ours_220_wandb_3_mix_noise_0.5 \
--run_name screw320 \
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
"${dataset_flags[@]}" mvtec $datapath




# python3 main_ours.py \
# --gpu 6 \
# --seed 0 \
# --log_group simplenet_mvtec \
# --log_project MVTecAD_Results \
# --results_path results_emaKD \
# --run_name run \
# net \
# -b wideresnet50 \
# -le layer2 \
# -le layer3 \
# --pretrain_embed_dimension 1536 \
# --target_embed_dimension 1536 \
# --patchsize 3 \
# --meta_epochs 40 \
# --embedding_size 256 \
# --gan_epochs 4 \
# --noise_std 0.015 \
# --dsc_hidden 1024 \
# --dsc_layers 2 \
# --dsc_margin .5 \
# --pre_proj 1 \
# dataset \
# --batch_size 8 \
# --resize 226 \
# --imagesize 224 \
# --augment \
# --brightness 0.1 \
# --contrast 0.1 \
# --saturation 0.1 \
# "${dataset_flags[@]}" mvtec $datapath
