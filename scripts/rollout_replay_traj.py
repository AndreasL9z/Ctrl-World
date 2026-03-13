
# from openpi.training import config as config_pi
# from openpi.policies import policy_config
# from openpi_client import image_tools
import numpy as np


from accelerate import Accelerator
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import key_board_control, get_fk_solution

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import wandb
import json
from decord import VideoReader, cpu
import swanlab
import mediapy
import sys
from scipy.spatial.transform import Rotation as R
from skimage.metrics import structural_similarity
from scipy.linalg import sqrtm

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_FID = True
except ImportError:
    HAS_FID = False

try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
    HAS_R3D = True
except ImportError:
    try:
        from torchvision.models.video import r3d_18  # older torchvision without Weights enum
        HAS_R3D = True
        R3D_18_Weights = None
    except ImportError:
        HAS_R3D = False
        R3D_18_Weights = None



class agent():
    def __init__(self,args):
          
        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # # load pi policy
        # if 'pi05' in args.policy_type:
        #     config = config_pi.get_config("pi05_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid' 
        # elif 'pi0fast' in args.policy_type:
        #     config = config_pi.get_config("pi0fast_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0fast_droid'
        # elif 'pi0' in args.policy_type:
        #     config = config_pi.get_config("pi0_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0_droid'
        # else:
        #     raise ValueError(f"Unknown policy type: {args.policy_type}")
        # self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # load ctrl-world model        
        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(args.val_model_path, weights_only=False))
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("load world model success")
        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None,:]
            self.state_p99 = np.array(data_stat['state_99'])[None,:]
        

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

    def get_traj_info(self, id, start_idx=0, steps=8):
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps
        annotation_path = f"{val_dataset_dir}/annotation/val/{id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno['action'])
            except:
                length = anno["video_length"]
        frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)

        # get action (cartesian states) and joint pos
        instruction = anno['texts'][0]
        car_action = np.array(anno['states'])  # Cartesian states: [xyz, euler, gripper]
        car_action = car_action[frames_ids]
        if 'joints' in anno:
            # DROID: has separate 'joints' field for joint angles
            joint_pos = np.array(anno['joints'])
        else:
            # LIBERO: no 'joints' field, 'states' contains cartesian states (xyz+euler+gripper)
            # Note: variable name 'joint_pos' is misleading for LIBERO - it's actually cartesian states
            joint_pos = np.array(anno['states'])  # Actually cartesian states for LIBERO
        joint_pos = joint_pos[frames_ids]

        # get videos
        video_dict =[]
        video_latent = []
        for id in range(len(anno['videos'])):
            video_path = anno['videos'][id]['video_path']
            video_path = f"{val_dataset_dir}/{video_path}"
            # load videos from all views
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(range(length)).numpy()
            true_video = true_video[frames_ids]
            video_dict.append(true_video)

            # encode video
            device = self.device
            true_video = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video.permute(0,3,1,2).to(device) / 255.0*2-1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                batch_size = 32
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
    
            video_latent.append(x)

        
        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None):
        args = self.args
        image_cond = video_latent_cond

        # action should be normed
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames+args.num_history, args.action_dim)


        # predict future frames
        with torch.no_grad():
            bsz = action_cond.shape[0]
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)           
            pipeline = self.model.pipeline
            
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height*3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)


        # decode ground truth video
        true_video = torch.stack(video_latent_true, dim=0) # (bsz, 8,32,32)
        decoded_video = []
        bsz,frame_num = true_video.shape[:2]
        true_video = true_video.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,true_video.shape[0],args.decode_chunk_size):
            chunk = true_video[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video = torch.cat(decoded_video,dim=0)
        true_video = true_video.reshape(bsz,frame_num,*true_video.shape[1:])
        true_video = ((true_video / 2.0 + 0.5).clamp(0, 1)*255)
        true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)

        # decode predicted video
        decoded_video = []
        bsz,frame_num = latents.shape[:2]
        x = latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,x.shape[0],args.decode_chunk_size):
            chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)

        # concatenate true videos and video
        videos_cat = np.concatenate([true_video,videos],axis=-3) # (3, 8, 256, 256, 3)
        videos_cat = np.concatenate([video for video in videos_cat],axis=-2).astype(np.uint8) 

        return videos_cat, true_video, videos, latents  # np.uint8:(3, 8, 128, 256, 3) or (3, 8, 192, 320, 3)


# ---------------------------------------------------------------------------
# Video Quality Metrics
# ---------------------------------------------------------------------------
class VideoMetrics:
    """
    Accumulates and computes video quality metrics across multiple trajectories:
      MSE, PSNR, SSIM, LPIPS, UIQI  – computed per-frame, averaged over all calls
      FID                            – Fréchet Inception Distance over all frames
      FVD                            – Fréchet Video Distance via R3D-18 features

    Usage:
        tracker = VideoMetrics(device)
        tracker.update(real_np, pred_np)   # (V, T, H, W, 3) uint8, call per step
        results = tracker.print_results()  # prints + returns dict
    """

    def __init__(self, device: torch.device, compute_fid: bool = True, compute_fvd: bool = True):
        self.device = device

        # per-step scalar accumulators
        self.mse_scores:   list[float] = []
        self.psnr_scores:  list[float] = []
        self.ssim_scores:  list[float] = []
        self.lpips_scores: list[float] = []
        self.uiqi_scores:  list[float] = []

        # LPIPS
        self.lpips_fn = None
        if HAS_LPIPS:
            self.lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()
            print("[VideoMetrics] LPIPS initialised.")
        else:
            print("[VideoMetrics] lpips not installed – LPIPS skipped.")

        # FID (torchmetrics)
        self.fid_metric = None
        if compute_fid and HAS_FID:
            self.fid_metric = FrechetInceptionDistance(feature=2048).to(device)
            print("[VideoMetrics] FID initialised.")
        elif compute_fid:
            print("[VideoMetrics] torchmetrics not installed – FID skipped.")

        # FVD backbone: R3D-18 (torchvision)
        self.r3d_model = None
        self.real_video_feats: list = []
        self.pred_video_feats: list = []
        if compute_fvd and HAS_R3D:
            try:
                self.r3d_model = r3d_18(weights=R3D_18_Weights.DEFAULT if R3D_18_Weights else None)
            except Exception:
                self.r3d_model = r3d_18(pretrained=True)
            self.r3d_model.fc = nn.Identity()
            self.r3d_model = self.r3d_model.to(device).eval()
            print("[VideoMetrics] FVD backbone (R3D-18) initialised.")
        elif compute_fvd:
            print("[VideoMetrics] torchvision video models not available – FVD skipped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, real_videos: np.ndarray, pred_videos: np.ndarray):
        """
        Update all metrics with one batch of ground-truth / predicted videos.

        Args:
            real_videos: uint8 ndarray of shape (V, T, H, W, 3)
            pred_videos: uint8 ndarray of shape (V, T, H, W, 3)
        """
        V, T, H, W, _ = real_videos.shape
        real_flat = real_videos.reshape(V * T, H, W, 3).astype(np.float32)
        pred_flat = pred_videos.reshape(V * T, H, W, 3).astype(np.float32)

        # --- MSE & PSNR ---
        diff = real_flat - pred_flat
        mse = float(np.mean(diff ** 2))
        self.mse_scores.append(mse)
        psnr = 20.0 * np.log10(255.0 / (np.sqrt(mse) + 1e-8)) if mse > 0 else float('inf')
        self.psnr_scores.append(float(psnr))

        # --- SSIM (per-frame average) ---
        ssim_vals = []
        for r, p in zip(real_flat.astype(np.uint8), pred_videos.reshape(V * T, H, W, 3)):
            try:
                s = structural_similarity(r, p, channel_axis=-1, data_range=255)
            except TypeError:   # scikit-image < 0.19
                s = structural_similarity(r, p, multichannel=True, data_range=255)
            ssim_vals.append(s)
        self.ssim_scores.append(float(np.mean(ssim_vals)))

        # --- LPIPS (per-frame average) ---
        if self.lpips_fn is not None:
            real_t = torch.from_numpy(real_flat).permute(0, 3, 1, 2).to(self.device) / 127.5 - 1.0
            pred_t = torch.from_numpy(pred_flat).permute(0, 3, 1, 2).to(self.device) / 127.5 - 1.0
            self.lpips_scores.append(float(self.lpips_fn(real_t, pred_t).mean().item()))

        # --- UIQI (per-frame average) ---
        uiqi_vals = [
            self._uiqi(real_flat[i].astype(np.float64), pred_flat[i].astype(np.float64))
            for i in range(len(real_flat))
        ]
        self.uiqi_scores.append(float(np.mean(uiqi_vals)))

        # --- FID feature update ---
        if self.fid_metric is not None:
            real_u8 = torch.from_numpy(real_flat.astype(np.uint8)).permute(0, 3, 1, 2).to(self.device)
            pred_u8 = torch.from_numpy(pred_flat.astype(np.uint8)).permute(0, 3, 1, 2).to(self.device)
            self.fid_metric.update(real_u8, real=True)
            self.fid_metric.update(pred_u8, real=False)

        # --- FVD feature update ---
        if self.r3d_model is not None:
            self._extract_fvd_features(real_videos, pred_videos)

    def get_results(self) -> dict:
        results = {}
        if self.mse_scores:
            results['MSE']  = float(np.mean(self.mse_scores))
            results['PSNR'] = float(np.mean(self.psnr_scores))
        if self.ssim_scores:
            results['SSIM'] = float(np.mean(self.ssim_scores))
        if self.lpips_scores:
            results['LPIPS'] = float(np.mean(self.lpips_scores))
        if self.uiqi_scores:
            results['UIQI'] = float(np.mean(self.uiqi_scores))
        if self.fid_metric is not None:
            try:
                results['FID'] = float(self.fid_metric.compute().item())
            except Exception as e:
                print(f"[VideoMetrics] FID compute() failed: {e}")
        fvd = self._compute_fvd()
        if fvd is not None:
            results['FVD'] = fvd
        return results

    def print_results(self) -> dict:
        results = self.get_results()
        print("\n" + "=" * 52)
        print("  Video Quality Metrics (aggregated over all steps)")
        print("=" * 52)
        labels = {
            'MSE':   'MSE   (↓)',
            'PSNR':  'PSNR  (↑, dB)',
            'SSIM':  'SSIM  (↑)',
            'LPIPS': 'LPIPS (↓)',
            'UIQI':  'UIQI  (↑)',
            'FID':   'FID   (↓)',
            'FVD':   'FVD   (↓)',
        }
        for key, label in labels.items():
            if key in results:
                print(f"  {label:<20s}: {results[key]:.4f}")
        print("=" * 52 + "\n")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _uiqi(self, real: np.ndarray, pred: np.ndarray) -> float:
        """Universal Image Quality Index for a single H×W×3 pair (float64)."""
        channel_qs = []
        for c in range(real.shape[-1]):
            r, p = real[..., c].ravel(), pred[..., c].ravel()
            mu_r, mu_p = r.mean(), p.mean()
            var_r,  var_p  = r.var(), p.var()
            cov_rp = np.mean((r - mu_r) * (p - mu_p))
            denom  = (var_r + var_p) * (mu_r ** 2 + mu_p ** 2)
            if denom < 1e-8:
                channel_qs.append(1.0 if np.allclose(r, p) else 0.0)
            else:
                channel_qs.append((4.0 * cov_rp * mu_r * mu_p) / denom)
        return float(np.mean(channel_qs))

    def _extract_fvd_features(self, real_videos: np.ndarray, pred_videos: np.ndarray):
        """Extract R3D-18 clip features (one feature vector per view per video)."""
        # R3D-18 Kinetics normalisation
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        std  = torch.tensor([0.22803,  0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)
        V, T, H, W, _ = real_videos.shape

        for v in range(V):
            for src_np, feat_list in [(real_videos[v], self.real_video_feats),
                                      (pred_videos[v], self.pred_video_feats)]:
                # (T, H, W, 3) → (1, 3, T, H, W)
                x = torch.from_numpy(src_np).permute(3, 0, 1, 2).float().to(self.device) / 255.0
                x = x.unsqueeze(0)
                x = (x - mean) / std
                # R3D-18 spatial input should be 112×112
                if H != 112 or W != 112:
                    B, C, T_, H_, W_ = x.shape
                    x = x.permute(0, 2, 1, 3, 4).reshape(B * T_, C, H_, W_)
                    x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
                    x = x.reshape(B, T_, C, 112, 112).permute(0, 2, 1, 3, 4)
                with torch.no_grad():
                    feat = self.r3d_model(x).cpu().numpy()  # (1, feat_dim)
                feat_list.append(feat)

    def _compute_fvd(self):
        """Compute Fréchet distance over accumulated R3D-18 features."""
        if not self.real_video_feats:
            return None
        real_f = np.concatenate(self.real_video_feats, axis=0)
        pred_f = np.concatenate(self.pred_video_feats, axis=0)
        mu_r, mu_p = real_f.mean(0), pred_f.mean(0)
        if len(real_f) > 1:
            sig_r = np.cov(real_f, rowvar=False)
            sig_p = np.cov(pred_f, rowvar=False)
        else:
            d = real_f.shape[1]
            sig_r = np.zeros((d, d))
            sig_p = np.zeros((d, d))
        diff = mu_r - mu_p
        covmean = sqrtm(sig_r @ sig_p)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(np.dot(diff, diff) + np.trace(sig_r + sig_p - 2 * covmean))


if __name__ == "__main__":
    from config import wm_args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--task_type', type=str, default='replay')
    parser.add_argument('--action_dim', type=int, default=None, help='Action dimension (7 or 8)')
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args
    
    args = merge_args(args, args_new)

    # create rollout agent
    Agent = agent(args)
    eval_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_tracker = VideoMetrics(Agent.device)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    print(f'rollout with {args.task_type}')


    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):
        # read ground truth trajectory informations
        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8))
        text_i = instruction
        print("text_i:",instruction, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])

        # create buffers and push first frames to history buffer
        predicted_latents = None
        video_to_save = []
        info_to_save = []
        his_cond = []
        his_joint = []
        his_eef = []
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
        for i in range(Agent.args.num_history*4):
            his_cond.append(first_latent)  # (1, 4, 72, 40)
            his_joint.append(joint_pos_gt[0:1])
            his_eef.append(eef_gt[0:1])

        # interact loop
        for i in range(interact_num):
            # ground truth video
            start_id = int(i*(pred_step-1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]
            
            # prepare input for policy
            joint_first = his_joint[-1][0]
            state_first = his_eef[-1][0]
            if i==0:
                video_first = [v[0] for v in video_dict]
            else:
                video_first = [v[-1] for v in video_dict_pred]
            assert joint_first.shape[-1] == joint_pos_gt.shape[-1], f"Expected joint_first dim {joint_pos_gt.shape[-1]}, got {joint_first.shape}"
            assert state_first.shape[-1] == eef_gt.shape[-1], f"Expected state_first dim {eef_gt.shape[-1]}, got {state_first.shape}"
            
            # forward policy
            print("################ policy forward ####################")
            # in the trajectory replay model, we use action recorded in trajetcory
            cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, action_dim)
            print("action", cartesian_pose[0]) # output action for debug
            print("action", cartesian_pose[-1]) # output action for debug
            
            print("################ world model forward ################")
            print(f'traj_id:{val_id_i}, interact step: {i}/{interact_num}')
            # retrive history cond and action cond
            history_idx = args.history_idx
            his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)  # (num_history, action_dim)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)
            assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
            assert action_cond.shape == (int(num_history+num_frames), args.action_dim), f"Expected action_cond shape ({int(num_history+num_frames)}, {args.action_dim}), got {action_cond.shape}"
            assert his_cond_input.shape == (1, int(num_history), 4, 72, 40), f"Expected his_cond_input shape (1, {int(num_history)}, 72, 40), got {his_cond_input.shape}"
            # forward world model
            videos_cat, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm(action_cond, video_latent_true, current_latent, his_cond=his_cond_input,text=text_i if Agent.args.text_cond else None)
            metrics_tracker.update(true_videos, video_dict_pred)

            print("################ record information ################")
            # push current step to history buffer
            his_eef.append(cartesian_pose[pred_step-1:pred_step])
            his_cond.append(torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
            if i == interact_num - 1:
                video_to_save.append(videos_cat)  # save all frames for the last interaction step
            else:
                video_to_save.append(videos_cat[:pred_step-1]) # last frame is the first frame of next step, so we remove it here
                
        
        # save rollout video and info with parameters
        video = np.concatenate(video_to_save, axis=0)
        task_name = args.task_name
        text_id = text_i.replace(' ', '_').replace(',', '').replace('.', '').replace('\'', '').replace('\"', '')[:30]
        videos_dir = args.val_model_path.split('/')[:-1]
        videos_dir = '/'.join(videos_dir)
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_video = f"{args.save_dir}/{task_name}/video/time_{uuid}_traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}.mp4"
        os.makedirs(os.path.dirname(filename_video), exist_ok=True)
        mediapy.write_video(filename_video, video, fps=4)
        print(f"Saving video to {filename_video}")
        print("##########################################################################")

    # ---- final aggregated metrics across all trajectories / steps ----
    final_metrics = metrics_tracker.print_results()
    metrics_save_path = f"{args.save_dir}/{args.task_name}/metrics_{eval_start_time}.json"
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_save_path}")


# CUDA_VISIBLE_DEVICES=0 python rollout_replay_traj.py
        
        
