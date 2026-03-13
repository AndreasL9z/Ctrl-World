"""
Extract latent representations for LIBERO dataset
Based on extract_latent.py but adapted for LIBERO data format
"""

import mediapy
import os
from diffusers.models import AutoencoderKLTemporalDecoder
import torch
import numpy as np
import json
from torch.utils.data import Dataset
import pandas as pd
from accelerate import Accelerator
from PIL import Image
import io


class EncodeLatentDataset(Dataset): 
    def __init__(self, old_path, new_path, svd_path, device, size=(192, 320), rgb_skip=3):
        """
        Args:
            old_path: Path to LIBERO dataset (e.g., /scr2/yusenluo/libero)
            new_path: Output path for processed dataset
            svd_path: Path to SVD VAE model
            device: Device for VAE
            size: Target video size (height, width)
            rgb_skip: Frame skip rate (e.g., 3 means keep every 3rd frame)
        """
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)

        # Load task descriptions from tasks.jsonl
        def load_json_file(file_path):
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        
        tasks_data = load_json_file(f'{old_path}/meta/tasks.jsonl')
        self.task_descriptions = {t['task_index']: t['task'] for t in tasks_data}
        print(f"Loaded {len(self.task_descriptions)} tasks")
        
        # Load episode metadata from episodes.jsonl
        self.data = load_json_file(f'{old_path}/meta/episodes.jsonl')
        print(f"Loaded {len(self.data)} episodes")

    def __len__(self):
        return len(self.data)

    def decode_image(self, img_dict):
        """Decode image from bytes"""
        img_bytes = img_dict['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)

    def __getitem__(self, idx):
        traj_data = self.data[idx]
        episode_id = traj_data['episode_index']
        instruction = traj_data['tasks'][0]
        chunk_id = int(episode_id / 1000)
        
        # Determine train/val split (e.g., every 100th episode is val)
        data_type = 'val' if episode_id % 100 == 99 else 'train'
        
        # Load parquet file
        file_path = f'{self.old_path}/data/chunk-{chunk_id:03d}/episode_{episode_id:06d}.parquet'
        df = pd.read_parquet(file_path)
        length = len(df)
        
        # Extract states and actions
        states = []
        actions = []
        for i in range(length):
            states.append(df['state'][i].tolist())  # 8-dim cartesian state
            actions.append(df['actions'][i].tolist())  # 7-dim action
        
        # Extract videos from both cameras
        agentview_frames = []
        wrist_frames = []
        
        for i in range(length):
            # Decode images
            agentview_img = self.decode_image(df['image'][i])
            wrist_img = self.decode_image(df['wrist_image'][i])
            
            agentview_frames.append(agentview_img)
            wrist_frames.append(wrist_img)
        
        agentview_video = np.array(agentview_frames)  # (T, H, W, 3)
        wrist_video = np.array(wrist_frames)
        
        # Process trajectory
        try:
            self.process_traj(
                videos=[agentview_video, wrist_video],
                states=states,
                actions=actions,
                instruction=instruction,
                save_root=self.new_path,
                episode_id=episode_id,
                data_type=data_type,
                size=self.size,
                rgb_skip=self.skip,
                device=self.vae.device
            )
        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
            return 0
    
        return 0

    def process_traj(self, videos, states, actions, instruction, save_root, 
                     episode_id=0, data_type='val', size=(192, 320), rgb_skip=3, device='cuda'):
        """
        Process one trajectory: resize videos, encode to latent, save annotations
        
        Args:
            videos: List of numpy arrays [agentview_video, wrist_video], each (T, H, W, 3)
                   LIBERO has 2 cameras, will be padded to 3 for training compatibility
            states: List of state arrays (T, 8)
            actions: List of action arrays (T, 7)
            instruction: Task description string
            save_root: Output directory
            episode_id: Episode ID
            data_type: 'train' or 'val'
            size: Target size (height, width)
            rgb_skip: Frame skip rate
            device: Device for VAE
            
        Note:
            Training code expects 3 camera views (DROID format):
            - latent shape: (T, 4, 72, 40) where 72 = 24*3 (3 views stacked)
            - LIBERO only has 2 views, so we add zero-padding for the 3rd view
        """
        latent_videos = []
        
        for video_id, video in enumerate(videos):
            # video: (T, H, W, 3), uint8
            # Flip video 0 (agentview) horizontally if it's mirrored
            if video_id == 0:
                video = video[:, :, ::-1, :]  # Flip horizontally (H, W, C)
                video = np.ascontiguousarray(video)  # Make array contiguous after flip
            
            # Convert to torch and normalize to [-1, 1]
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1  # (T, 3, H, W)
            frames = frames[::rgb_skip]  # Skip frames
            
            # Resize
            x = torch.nn.functional.interpolate(
                frames, size=size, mode='bilinear', align_corners=False
            )  # (T', 3, H', W')
            
            # Save resized video
            resize_video = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            os.makedirs(f"{save_root}/videos/{data_type}/{episode_id}", exist_ok=True)
            mediapy.write_video(
                f"{save_root}/videos/{data_type}/{episode_id}/{video_id}.mp4", 
                resize_video, fps=5
            )
            
            # Encode to latent using SVD VAE
            x = x.to(device)
            with torch.no_grad():
                batch_size = 64
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = self.vae.encode(batch).latent_dist.sample()
                    latent = latent.mul_(self.vae.config.scaling_factor).cpu()
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            
            # Save latent
            os.makedirs(f"{save_root}/latent_videos/{data_type}/{episode_id}", exist_ok=True)
            torch.save(x, f"{save_root}/latent_videos/{data_type}/{episode_id}/{video_id}.pt")
            latent_videos.append(x)
        
        # LIBERO only has 2 cameras, add a zero-padded third view for compatibility with training code
        # Training expects 3 views vertically stacked (height=72 = 24*3)
        if len(videos) == 2:
            # Create zero-filled latent with same shape as other views
            zero_latent = torch.zeros_like(latent_videos[0])
            zero_video = np.zeros_like(resize_video, dtype=np.uint8)
            
            # Save zero-padded video
            os.makedirs(f"{save_root}/videos/{data_type}/{episode_id}", exist_ok=True)
            mediapy.write_video(
                f"{save_root}/videos/{data_type}/{episode_id}/2.mp4", 
                zero_video, fps=5
            )
            
            # Save zero-padded latent
            os.makedirs(f"{save_root}/latent_videos/{data_type}/{episode_id}", exist_ok=True)
            torch.save(zero_latent, f"{save_root}/latent_videos/{data_type}/{episode_id}/2.pt")
            latent_videos.append(zero_latent)
        
        # Align states with video frames (apply same skip)
        # states: (T, 8) -> 8-dim cartesian state [xyz(3), euler(3), gripper_left(1), gripper_right(1)]
        states_array = np.array(states)  # (T, 8)
        states_aligned = states_array[::rgb_skip].tolist()  # Apply skip
        
        # Create annotation file
        info = {
            "texts": [instruction],
            "episode_id": int(episode_id),
            "video_length": frames.shape[0],  # After skipping
            "state_length": len(states_aligned),
            "raw_length": len(states),
            "videos": [
                {"video_path": f"videos/{data_type}/{episode_id}/0.mp4"},  # agentview
                {"video_path": f"videos/{data_type}/{episode_id}/1.mp4"},  # wrist
                {"video_path": f"videos/{data_type}/{episode_id}/2.mp4"},  # placeholder (zero-padded)
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{episode_id}/0.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{episode_id}/1.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{episode_id}/2.pt"},  # placeholder (zero-padded)
            ],
            'states': states_aligned,  # 8-dim cartesian state (aligned with video frames)
            'observation.state': states,  # 8-dim LIBERO cartesian state [xyz, euler, gripper_left, gripper_right]
            'actions': actions,  # 7-dim LIBERO action
        }
        
        os.makedirs(f"{save_root}/annotation/{data_type}", exist_ok=True)
        with open(f"{save_root}/annotation/{data_type}/{episode_id}.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--libero_data_path', type=str, 
                       default='/scr2/yusenluo/libero',
                       help='Path to LIBERO dataset (should contain data/ and meta/ folders)')
    parser.add_argument('--output_path', type=str, 
                       default='dataset_example/libero',
                       help='Output path for processed dataset')
    parser.add_argument('--svd_path', type=str, 
                       default='models/svd',
                       help='Path to SVD model')
    parser.add_argument('--size', type=int, nargs=2, default=[192, 320],
                       help='Target video size (height width) - must be [192, 320] for latent size [24, 40]')
    parser.add_argument('--rgb_skip', type=int, default=3,
                       help='Frame skip rate (e.g., 3 means keep every 3rd frame, 10Hz->~3Hz)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (process only 5 episodes)')
    args = parser.parse_args()

    accelerator = Accelerator()
    
    dataset = EncodeLatentDataset(
        old_path=args.libero_data_path,
        new_path=args.output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        size=tuple(args.size),
        rgb_skip=args.rgb_skip,
    )
    
    tmp_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    
    for idx, _ in enumerate(tmp_data_loader):
        if args.debug and idx == 5:
            break
        if idx % 100 == 0 and accelerator.is_main_process:
            print(f"Processed {idx} episodes")

    print("Done!")

# Run command:
# accelerate launch dataset_example/extract_latent_libero_new.py \
#     --libero_data_path /scr2/yusenluo/libero \
#     --output_path dataset_example/libero \
#     --svd_path models/svd \
#     --size 192 320 \
#     --rgb_skip 3
