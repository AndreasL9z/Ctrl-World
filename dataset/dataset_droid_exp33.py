import json
import os
import random
import warnings
import traceback
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import imageio
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange
from scipy.spatial.transform import Rotation as R  
import decord

class Dataset_mix(Dataset):
    def __init__(
            self,
            args,
            mode = 'val',
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode

        # dataset stucture
        # dataset_root_path/dataset_name/annotation_name/mode/traj
        # dataset_root_path/dataset_name/video/mode/traj
        # dataset_root_path/dataset_name/latent_video/mode/traj

        # samples:{'ann_file':xxx, 'frame_idx':xxx, 'dataset_name':xxx}

        # prepare all datasets path
        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []


        dataset_root_path = args.dataset_root_path
        dataset_names = args.dataset_names.split('+')
        dataset_meta_info_path = args.dataset_meta_info_path
        dataset_cfgs = args.dataset_cfgs.split('+')
        self.prob = args.prob
        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            data_json_path = f'{dataset_meta_info_path}/{dataset_cfg}/{mode}_sample.json'
     
            with open(data_json_path, "r") as f:
                samples = json.load(f)
            dataset_path = [os.path.join(dataset_root_path, dataset_name) for sample in samples]
            print(f"ALL dataset, {len(samples)} samples in total")
            self.dataset_path_all.append(dataset_path)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))

            # prepare normalization
            with open(f'{dataset_meta_info_path}/{dataset_name}/stat.json', "r") as f:
                data_stat = json.load(f)
                state_p01 = np.array(data_stat['state_01'])[None,:]
                state_p99 = np.array(data_stat['state_99'])[None,:]
                self.norm_all.append((state_p01, state_p99))
        
        self.max_id = max(self.samples_len)
        print('samples_len:',self.samples_len, 'max_id:',self.max_id)

    def __len__(self):
        return self.max_id

    def _load_latent_video(self, video_path, frame_ids):
        with open(video_path,'rb') as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
        max_frames = video_tensor.size()[0]
        frame_ids =  [int(frame_id) if frame_id < max_frames else max_frames-1 for frame_id in frame_ids]
        frame_data = video_tensor[frame_ids]
        return frame_data

    def _get_frames(self, label, frame_ids, cam_id, pre_encode, video_dir, use_img_cond=False):
        # directly load videos latent after svd-vae encoder
        assert cam_id is not None
        assert pre_encode == True
        if pre_encode: 
            video_path = label['latent_videos'][cam_id]['latent_video_path']
            video_path = os.path.join(video_dir,video_path)
            try:
                frames = self._load_latent_video(video_path, frame_ids)
            except:
                video_path = video_path.replace("latent_videos", "latent_videos_svd")
                frames = self._load_latent_video(video_path, frame_ids)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode, video_dir):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id = temp_cam_id, pre_encode = pre_encode, video_dir=video_dir)
        return frames, temp_cam_id

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata

    def __getitem__(self, index):

        # first sample the dataset id, than sample the data from the dataset
        dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]
        dataset_dir = dataset_path[index]

        # get annotation
        frame_ids = sample['frame_ids']
        ann_file = f'{dataset_dir}/{self.args.annotation_name}/{self.mode}/{sample["episode_id"]}.json'
        with open(ann_file, "r") as f:
            label = json.load(f)
            
        # since we downsample the video from 15hz to 5 hz to save the storage space, the frame id is 1/3 of the state id
        # For DROID: use observation.state.joint_position length
        # For LIBERO: use the 'states' or 'observation.state' length directly
        if 'observation.state.joint_position' in label:
            # DROID: original state length (before downsampling)
            joint_len = len(label['observation.state.joint_position'])-1
            frame_len = np.floor(joint_len / 3)
        elif 'states' in label:
            # LIBERO: already downsampled states, video_length gives frame count
            frame_len = label['video_length'] - 1
        else:
            raise ValueError("Cannot determine trajectory length from annotation")
        skip = random.randint(1, 2)
        skip_his = int(skip*4)
        p = random.random()
        if p < 0.15:
            skip_his = 0
        
        # rgb_id and state_id
        frame_now = frame_ids[0]
        rgb_id = []
        for i in range(self.args.num_history,0,-1):
            rgb_id.append(int(frame_now - i*skip_his))
        rgb_id.append(frame_now)
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i*skip))
        rgb_id = np.array(rgb_id)
        rgb_id = np.clip(rgb_id, 0, frame_len).tolist()
        rgb_id = [int(frame_id) for frame_id in rgb_id]
        state_id = np.array(rgb_id)*self.args.down_sample


        # prepare data
        data = dict()

        # instructions
        data['text'] = label['texts'][0]

        # stack tokens of multi-view
        cond_cam_id1 = 0
        cond_cam_id2 = 1
        cond_cam_id3 = 2
        latnt_cond1,_ = self._get_obs(label, rgb_id, cond_cam_id1, pre_encode=True, video_dir=dataset_dir)
        latnt_cond2,_ = self._get_obs(label, rgb_id, cond_cam_id2, pre_encode=True, video_dir=dataset_dir)
        latnt_cond3,_ = self._get_obs(label, rgb_id, cond_cam_id3, pre_encode=True, video_dir=dataset_dir)
        latent = torch.zeros((self.args.num_frames+self.args.num_history, 4, 72, 40), dtype=torch.float32)
        latent[:,:,0:24] =  latnt_cond1
        latent[:,:,24:48] = latnt_cond2
        latent[:,:,48:72] = latnt_cond3
        data['latent'] = latent.float()

        # =================================================================
        # Seg-mask spatial weight（用于训练 loss 的前景区域加权）
        #
        # 背景：seg mask 文件保存在 {seg_root_path}/{train|val}/{episode_id}/{cam_id}.pt
        #       每个文件 shape = [T_seg, C, 192, 320]，dtype=float32，值域 {0, 1}
        #       C = 该帧检测到的 object 数量（各轨迹不同）
        #
        # 时间对齐说明：
        #   seg 是从原始视频（raw_length 帧）以约 stride=5 计算的，
        #   而视频帧（video_length）是以约 stride=2 从原始帧采样的，
        #   因此 T_seg ≈ video_length / 2.5，二者帧率不同。
        #   映射公式：seg_idx = round(video_frame_idx * T_seg / video_length)
        #
        # 空间对齐说明：
        #   seg 与视频同为 192×320 分辨率，VAE 将 192×320 → 24×40（8× 下采样）。
        #   因此把 seg 做同等 bilinear 下采样到 24×40 即可与 latent 空间对齐。
        #
        # 返回：data['seg_weight'] shape = [T_total, 1, 72, 40]
        #   72 = 3 相机各 24 行垂直拼接，与 latent 的 H 维度严格对应：
        #     cam0 → rows [0:24], cam1 → rows [24:48], cam2 → rows [48:72]
        #   值域 [0, 1]，前景=1，背景=0；
        #   在 model 的 loss 中会转为 1.0 + alpha * seg_weight 作为权重。
        # =================================================================
        T_total = self.args.num_history + self.args.num_frames
        seg_root = getattr(self.args, 'seg_root_path', None)

        if seg_root is not None:
            # 获取视频总帧数，用于帧索引映射
            # LIBERO annotation 含 'video_length'，DROID 不含（DROID 无 seg 文件，会走 except）
            video_length = label.get('video_length', max(rgb_id) + 1)

            cam_masks = []  # 存储 3 个相机各自下采样后的 mask
            for cam_id in [0, 1, 2]:
                # LIBERO 只有两路真实相机（cam0=agentview, cam1=wrist）
                # cam2 在 latent 中是全黑零填充（见 extract_latent_libero_new.py），
                # 对应位置没有真实图像内容，不应施加任何前景权重。
                # 虽然 seg2 目录里存在 2.pt（内容是 cam0 的副本），但直接忽略，
                # 强制置零以保证与 latent 的语义一致。
                if cam_id == 2:
                    cam_masks.append(torch.zeros(T_total, 1, 24, 40))
                    continue

                seg_path = os.path.join(
                    seg_root, self.mode, str(sample['episode_id']), f'{cam_id}.pt'
                )
                try:
                    # 加载预计算的 seg mask：[T_seg, C, 192, 320]
                    seg = torch.load(seg_path, map_location='cpu')
                    T_seg = seg.shape[0]

                    # 将 rgb_id（视频帧索引）映射到对应的 seg 帧索引
                    # 公式：seg_idx = round(video_frame_idx * T_seg / video_length)
                    # clamp 到合法范围 [0, T_seg-1]
                    seg_ids = [
                        min(round(f * T_seg / video_length), T_seg - 1)
                        for f in rgb_id
                    ]

                    # 按映射后的帧索引取出对应帧：[T_total, C, 192, 320]
                    seg_frames = seg[seg_ids]

                    # 对所有 object 通道取 union（max）→ 单通道前景 mask
                    # shape: [T_total, 1, 192, 320]，值域 {0.0, 1.0}
                    mask = seg_frames.max(dim=1, keepdim=True).values.float()

                    # Bilinear 下采样到 latent 分辨率 24×40
                    # 与 VAE 的 8× 空间下采样等效，保证位置一一对应
                    # shape: [T_total, 1, 24, 40]
                    mask = torch.nn.functional.interpolate(
                        mask, size=(24, 40), mode='bilinear', align_corners=False
                    )

                except Exception:
                    # 文件不存在（DROID 无 seg）或读取出错 → 该相机用零 mask（均匀权重）
                    mask = torch.zeros(T_total, 1, 24, 40)

                cam_masks.append(mask)

            # 沿 H 维度拼接 3 个相机 mask，与 latent 的空间布局对齐
            # [T_total, 1, 24, 40] × 3  →  [T_total, 1, 72, 40]
            seg_weight = torch.cat(cam_masks, dim=2)
        else:
            # seg_root_path 未配置 → 全零 mask（等价于原始均匀权重 MSE，无任何改变）
            seg_weight = torch.zeros(T_total, 1, 72, 40)

        # shape: [T_total, 1, 72, 40]，值域 [0, 1]
        data['seg_weight'] = seg_weight

        # =================================================================
        # Optical flow for temporal warping consistency loss
        #
        # File: {flow_root_path}/{train|val}/{episode_id}/{cam_id}.pt
        #   shape = [T_flow, 2, 256, 256], dtype=float32
        #   channel 0 = dx (horizontal), channel 1 = dy (vertical), unit: pixels at 256x256
        #   flow[t] = pixel displacement from original frame t to t+1
        #
        # Only compute warp loss between consecutive FUTURE frame pairs:
        #   (future[0]->future[1]), (future[1]->future[2]), ... (num_frames-1 pairs)
        # When skip>1, accumulate multiple flow fields (linear approximation).
        #
        # Temporal mapping (same as seg):
        #   flow_idx = round(video_frame_idx * T_flow / video_length)
        #
        # Spatial: flow 256x256 -> latent 24x40
        #   bilinear downsample + value rescaling: dx *= 40/256, dy *= 24/256
        #   (video was squished from 192x320 to 256x256 for flow computation)
        #
        # Returns: data['flow'] shape = [num_frames-1, 2, 72, 40]
        #   cam0->H[0:24], cam1->H[24:48], cam2->H[48:72] (zeros)
        # =================================================================
        T_future_pairs = self.args.num_frames - 1
        flow_root = getattr(self.args, 'flow_root_path', None)

        if flow_root is not None and T_future_pairs > 0:
            video_length_f = label.get('video_length', max(rgb_id) + 1)
            future_rgb_ids = rgb_id[self.args.num_history:]

            cam_flows = []
            for cam_id in [0, 1, 2]:
                if cam_id == 2:
                    cam_flows.append(torch.zeros(T_future_pairs, 2, 24, 40))
                    continue

                flow_path = os.path.join(
                    flow_root, self.mode, str(sample['episode_id']), f'{cam_id}.pt'
                )
                try:
                    flow_data = torch.load(flow_path, map_location='cpu')  # (T_flow, 2, 256, 256)
                    T_flow = flow_data.shape[0]

                    pair_flows = []
                    for t in range(T_future_pairs):
                        f_start = future_rgb_ids[t]
                        f_end   = future_rgb_ids[t + 1]

                        # Map video-frame indices to subsampled-frame indices
                        # (seg/flow have stride ~2.5x relative to video frames)
                        flow_idx_start = min(round(f_start * T_flow / video_length_f), T_flow - 1)
                        flow_idx_end   = min(round(f_end   * T_flow / video_length_f), T_flow - 1)
                        # Use subsampled-frame difference as accumulation steps,
                        # not video-frame difference (which would overcount).
                        # max(..., 1): when both map to the same subsampled frame,
                        # still use that one flow field as the best approximation.
                        n_sub_steps = max(flow_idx_end - flow_idx_start, 1)

                        # Accumulate n_sub_steps consecutive flow fields
                        accumulated = torch.zeros(2, 256, 256)
                        for s in range(n_sub_steps):
                            fi = min(flow_idx_start + s, T_flow - 1)
                            accumulated += flow_data[fi]

                        # Bilinear downsample 256x256 -> 24x40
                        acc_small = torch.nn.functional.interpolate(
                            accumulated.unsqueeze(0), size=(24, 40),
                            mode='bilinear', align_corners=False
                        ).squeeze(0)  # (2, 24, 40)

                        # Rescale flow values from 256px units to latent-px units
                        acc_small[0] *= (40.0 / 256.0)  # dx: width direction
                        acc_small[1] *= (24.0 / 256.0)  # dy: height direction

                        pair_flows.append(acc_small)

                    cam_flows.append(torch.stack(pair_flows, dim=0))  # (T_future_pairs, 2, 24, 40)

                except Exception:
                    cam_flows.append(torch.zeros(T_future_pairs, 2, 24, 40))

            # Concat along H: (T_future_pairs, 2, 24, 40) x3 -> (T_future_pairs, 2, 72, 40)
            data['flow'] = torch.cat(cam_flows, dim=2)
        else:
            data['flow'] = torch.zeros(T_future_pairs, 2, 72, 40)

        # prepare action cond data
        # Support both DROID format (observation.state.cartesian_position + gripper_position)
        # and LIBERO format (states field directly)
        if 'observation.state.cartesian_position' in label:
            # DROID format: 6D cartesian + 1D gripper = 7D
            cartesian_pose = np.array(label['observation.state.cartesian_position'])[state_id]
            gripper_pose = np.array(label['observation.state.gripper_position'])[state_id][..., np.newaxis]
            action = np.concatenate((cartesian_pose, gripper_pose), axis=-1)
        elif 'states' in label:
            # LIBERO format: 8D states (xyz + euler + gripper_left + gripper_right)
            # Use the downsampled 'states' field which already matches video frame rate
            action = np.array(label['states'])[state_id]
        else:
            raise ValueError("Annotation must contain either 'observation.state.cartesian_position' (DROID) or 'states' (LIBERO)")
        
        action = self.normalize_bound(action, state_p01, state_p99)
        data['action'] = torch.tensor(action).float()

        return data
        

if __name__ == "__main__":

    from config import wm_args
    args = wm_args()
    train_dataset = Dataset_mix(args,mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    for data in tqdm(train_loader,total=len(train_loader)):
        print(data['ann_file'])

    