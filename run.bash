#!/bin/bash
#SBATCH --job-name=ctrl-world    # Job name
#SBATCH --output=/scr2/yusenluo/Ctrl-World/train_wm_libero_seq_3_seg_loss_flow_loss_new.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                     # Number of GPUs                
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --time=24:00:00
#SBATCH --mem=64G

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate ctrl-world

# CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
#   --dataset_root_path dataset_example \
#   --dataset_meta_info_path dataset_meta_info \
#   --dataset_names libero \
#   --svd_model_path models/svd \
#   --clip_model_path models/clip-vit-base-patch32 \
#   --ckpt_path model_ckpt/libero_seq_3_seg_loss_flow_loss/checkpoint-35000.pt \
#   --action_dim 8 \
#   --task_type replay_libero
#   > /scr2/yusenluo/Ctrl-World/rollout_replay_libero_seq_3_seg_loss_flow_loss_new.txt


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py \
#     --task_type interact_libero \
#     --dataset_root_path dataset_example \
#     --dataset_meta_info_path dataset_meta_info \
#     --dataset_names libero \
#     --svd_model_path models/svd \
#     --clip_model_path models/clip-vit-base-patch32 \
#     --ckpt_path model_ckpt/libero_seq_3/checkpoint-40000.pt 




# # python dataset_meta_info/create_meta_info.py --droid_output_path dataset_example/libero --dataset_name libero


accelerate launch --main_process_port 29501 --mixed_precision fp16 scripts/train_wm.py \
    --dataset_root_path dataset_example \
    --dataset_meta_info_path dataset_meta_info \
    --dataset_names libero \
    --svd_model_path models/svd \
    --clip_model_path models/clip-vit-base-patch32 \
    --action_dim 8 \
    --down_sample 1 \
    --seg_root_path /scr/shared/world_model/libero_seg2_stride2 \
    --seg_loss_alpha 3.0 \
    --flow_root_path /scr/shared/world_model/libero_optical_flow_stride2 \
    --flow_loss_lambda 0.2 \
    --num_train_epochs 5 \
    --validation_steps 2000


# # CUDA_VISIBLE_DEVICES=0 accelerate launch dataset_example/extract_latent_libero_new.py \
# #     --libero_data_path /scr2/yusenluo/libero \
# #     --output_path dataset_example/libero \
# #     --svd_path models/svd \
# #     --rgb_skip 2 \
# #     --size 192 320

# Train World Model on LIBERO dataset
# accelerate launch --main_process_port 29501 scripts/train_wm.py \
#     --dataset_root_path dataset_example \
#     --dataset_meta_info_path dataset_meta_info \
#     --dataset_names libero \
#     --svd_model_path models/svd \
#     --clip_model_path models/clip-vit-base-patch32 \
#     --action_dim 8 \
#     --down_sample 1 