#!/bin/bash
#SBATCH --job-name=ctrl-world    # Job name
#SBATCH --output=/scr2/yusenluo/Ctrl-World/rollout_replay_traj.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=gpu:1                     # Number of GPUs                
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --time=48:00:00
#SBATCH --mem=64G

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh
conda activate ctrl-world

CUDA_VISIBLE_DEVICES=6 python scripts/rollout_replay_traj.py \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names libero \
  --svd_model_path models/svd \
  --clip_model_path models/clip-vit-base-patch32 \
  --ckpt_path model_ckpt/libero/checkpoint-18000.pt \
  --action_dim 8 \
  --task_type replay_libero 


CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py 
    --task_type interact_libero \
    --dataset_root_path dataset_example \
    --dataset_meta_info_path dataset_meta_info \
    --dataset_names libero \
    --svd_model_path models/svd \
    --clip_model_path models/clip-vit-base-patch32 \
    --ckpt_path model_ckpt/libero/checkpoint-18000.pt \
    --pi_ckpt model_ckpt/libero/checkpoint-18000.pt \
    --task_type interact_libero


# accelerate launch dataset_example/extract_latent_libero.py \
#     --frame_skip 4 \
#     --flip_horizontal


# python dataset_meta_info/create_meta_info.py --droid_output_path dataset_example/libero --dataset_name libero


# accelerate launch --main_process_port 29501 scripts/train_wm.py \
#     --dataset_root_path dataset_example \
#     --dataset_meta_info_path dataset_meta_info \
#     --dataset_names libero \
#     --svd_model_path models/svd \
#     --clip_model_path models/clip-vit-base-patch32 \
#     --action_dim 8 \
#     --down_sample 1