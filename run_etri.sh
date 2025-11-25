datapath="/workspace/Downloads/17.ETRI_data/printingplate/data1" 
datasets=('etri_plate')

dataset_flags=("-d" "${datasets[0]}")

## baseline run printing plate

# python3 main.py \
# --gpu 0 \
# --seed 0 \
# --log_group simplenet_etri \
# --log_project ETRI_Results \
# --results_path results_ETRI_baseline_printing \
# --run_name ETRI_Baseline \
# net \
# -b wideresnet50 \
# -le layer2 \
# -le layer3 \
# --pretrain_embed_dimension 1536 \
# --target_embed_dimension 1536 \
# --patchsize 3 \
# --meta_epochs 10 \
# --embedding_size 256 \
# --gan_epochs 4 \
# --noise_std 0.015 \
# --dsc_hidden 1024 \
# --dsc_layers 2 \
# --dsc_margin .5 \
# --pre_proj 1 \
# dataset \
# --batch_size 64 \
# --resize 256 \
# --imagesize 224 \
# --augment \
# --brightness 0.0 \
# --contrast 0.0 \
# --saturation 0.0 \
# "${dataset_flags[@]}" etri_printing "$datapath"


## SimpleNet++ run printing plate

python3 main_ours.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_etri \
--log_project ETRI_Results \
--results_path results_ETRI_ours_printing \
--run_name ETRI_Ours \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 10 \
--embedding_size 256 \
--gan_epochs 4 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 64 \
--resize 256 \
--imagesize 224 \
--augment \
--brightness 0.1 \
--contrast 0.1 \
--saturation 0.1 \
"${dataset_flags[@]}" etri_printing "$datapath"


## baseline run color plate

# python3 main.py \
# --gpu 0 \
# --seed 0 \
# --log_group simplenet_etri \
# --log_project ETRI_Results \
# --results_path results_ETRI_baseline_color \
# --run_name ETRI_Baseline \
# net \
# -b wideresnet50 \
# -le layer2 \
# -le layer3 \
# --pretrain_embed_dimension 1536 \
# --target_embed_dimension 1536 \
# --patchsize 3 \
# --meta_epochs 10 \
# --embedding_size 256 \
# --gan_epochs 4 \
# --noise_std 0.015 \
# --dsc_hidden 1024 \
# --dsc_layers 2 \
# --dsc_margin .5 \
# --pre_proj 1 \
# dataset \
# --batch_size 64 \
# --resize 256 \
# --imagesize 224 \
# --augment \
# --brightness 0.0 \
# --contrast 0.0 \
# --saturation 0.0 \
# "${dataset_flags[@]}" etri "$datapath"


## SimpleNet++ run color plate

# python3 main_ours.py \
# --gpu 0 \
# --seed 0 \
# --log_group simplenet_etri \
# --log_project ETRI_Results \
# --results_path results_ETRI_ours_color \
# --run_name ETRI_Ours \
# net \
# -b wideresnet50 \
# -le layer2 \
# -le layer3 \
# --pretrain_embed_dimension 1536 \
# --target_embed_dimension 1536 \
# --patchsize 3 \
# --meta_epochs 10 \
# --embedding_size 256 \
# --gan_epochs 4 \
# --noise_std 0.015 \
# --dsc_hidden 1024 \
# --dsc_layers 2 \
# --dsc_margin .5 \
# --pre_proj 1 \
# dataset \
# --batch_size 64 \
# --resize 256 \
# --imagesize 224 \
# --augment \
# --brightness 0.1 \
# --contrast 0.1 \
# --saturation 0.1 \
# "${dataset_flags[@]}" etri "$datapath"


