#!/usr/bin/env python
"""
Offline rollout on a LeRobot dataset episode from the training/validation set.
Simulates a rollout by feeding observations to the model and collecting predicted actions
from a particular episode in the dataset. Compares predicted actions to ground-truth actions.
"""
import argparse
import json
import copy
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

import openpi.path_robotics_robot_learning.path_robotics_dataset as path_robotics_dataset
from openpi.path_robotics_robot_learning.path_robotics_utils import ActionQueue
from openpi.training import config as _config
from openpi.policies import policy_config

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
    module=r"torchvision\.io\._video_deprecation_warning",
)


os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/hf_datasets_cache")
print(os.getenv("HF_DATASETS_CACHE"))

# ---------- Plotting Code --------------
def plot_2x8_dims(preds, gts, out_path="plots/ep_2x8.png", dims=None, title="", sharey=False):
    """
    Plot 16 per-dimension time-series as a 2x8 grid: GT vs Pred.

    Args:
        preds, gts: numpy arrays of shape [T, A]
        out_path: output PNG path
        dims: list of dimension indices to plot (<=16). Defaults to first 16.
        title: figure title
        sharey: share y-axis across subplots
    """
    if preds.shape != gts.shape:
        raise ValueError(f"Shape mismatch: preds{preds.shape} vs gts{gts.shape}")
    T, A = preds.shape  # noqa: N806

    dims = list(range(min(16, A))) if dims is None else [int(d) for d in dims if 0 <= int(d) < A][:16]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(2, 8, figsize=(20, 6), sharex=True, sharey=sharey)
    axes = axes.ravel()
    t = np.arange(T)

    # Plot requested dims (up to 16); turn off extra axes
    for i in range(16):
        ax = axes[i]
        if i < len(dims):
            d = dims[i]
            ax.plot(t, gts[:, d], label="gt", linewidth=1.2)
            ax.plot(t, preds[:, d], label="pred", linewidth=1.2, linestyle="--")
            ax.set_title(f"dim {d}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right", frameon=False, fontsize=9)
        else:
            ax.axis("off")

        ax.set_ylim(-2, 2)

    for ax in axes[8:]:
        ax.set_xlabel("frame")
    if title:
        fig.suptitle(title, y=0.98, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved 2x8 plot to {out_path}")


# ---------- main offline roll ----------
def offline_rollout(
        
    ckpt_path: str,
    config_name:str,
    root: Path,
    episode_index: int,
    save_preds: str | None,
    instruction: str | None,
    override_fps: int | None,
    n_step_actions: int,
):
    """
    Main function for Offline rollout on a LeRobot dataset episode from the training/validation set.
    """
    # model + cfg
    config = _config.get_config(config_name)
    print("Config:", config, "\n")

    policy = policy_config.create_trained_policy(config, ckpt_path)
    print("Policy created:", policy, "\n")


    # stats
    eps_stats = root / "meta" / "episodes_stats.jsonl"
    if not eps_stats.is_file():
        raise FileNotFoundError(f"Missing stats file: {eps_stats}")

    print("Dataset statistics computed successfully.\n")

    # dataset
    meta = path_robotics_dataset.RobotLearningDatasetMetadata(repo_id="", root=root)
    episodes = list(meta.episodes.keys())
    test_episodes = episodes[int(len(episodes) * 0.0) :]
    print("Fps:", override_fps, "\n")

    ds = path_robotics_dataset.RobotLearningDataset("", root=root, episodes=test_episodes, valid_ratio=0)

    ep_idx = np.asarray(ds.hf_dataset["episode_index"])
    frame_ids = np.where(ep_idx == episode_index)[0]
    if frame_ids.size == 0:
        raise ValueError(f"Episode {episode_index} not found. Available range might be 0..{ep_idx.max()}")

    preds, gts, inference_times = [], [], []

    # Skip frames to account for override fps
    factor = ds.meta.fps / override_fps if override_fps is not None else 1.0
    print("FPS factor:", factor)

    # Initialize ActionQueue to allow for open-loop chunk-execution (not strictly necessary here)
    rtc_enabled = True  # Set to True to simulate RTC behavior
    action_queue = ActionQueue(rtc_enabled=rtc_enabled)

    for step, i in enumerate(frame_ids):
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1})")
        
        start_time = time.time()

        if factor != 1.0 and step % int(factor) != 0:
            continue

        sample = ds[int(i)]
        sample_gt = copy.deepcopy(sample)

        sample["task"] = instruction
        for k, v in list(sample.items()):
            if k.startswith("observation.") and torch.is_tensor(v) and v.ndim >= 1:
                sample[k] = v.unsqueeze(0)
            elif k.startswith("action") and torch.is_tensor(v) and v.ndim >= 1:
                sample[k] = v.unsqueeze(0)

        if step == 0:
            # Warmup (compilation)
            _ = policy.infer(sample)

        if step == 0:
            op = policy.infer(sample)
            pred = op["actions"]  # [1, n_step, A]
            
            action_queue.merge(
                original_actions=pred,
                processed_actions=pred,    
                real_delay=0,
                action_index_before_inference=0,
            )
        elif action_queue.qsize() == (config.model.action_horizon - n_step_actions):
            op = policy.infer(sample)
            pred = op["actions"]  # [1, n_step, A]

            action_queue.merge(
                original_actions=pred,
                processed_actions=pred,    
                real_delay=0,
                action_index_before_inference=n_step_actions,
            )
        
        predicted_action = action_queue.get()
    
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        gt = sample_gt["actions"].numpy()
        preds.append(predicted_action.copy())
        gts.append(gt.copy())
    
    
    preds = np.stack(preds)  # [T,A]
    gts = np.stack(gts)  # [T,A]
    inference_times = np.array(inference_times)
    mae = np.mean(np.abs(preds - gts), axis=0)
    rmse = np.sqrt(np.mean((preds - gts) ** 2, axis=0))
    print(f"[offline] episode {episode_index}")
    print(f"  MAE (mean over dims):  {mae.mean():.6f}")
    print(f"  RMSE(mean over dims):  {rmse.mean():.6f}")
    print(f"  MAE first 8 dims: {np.round(mae[:8], 5)}")
    print()
    print(f"  Inference time (mean ± std): {inference_times.mean()} ± {inference_times.std()} s")

    plot_2x8_dims(
        preds, gts, out_path=f"plots/ep{episode_index}_2x8.png", title=f"Episode {episode_index}"
    )

    # # Save the Preds
    # if save_preds:
    #     preds_path = save_preds + f"/ep{episode_index}_preds.npy"
    #     preds_path = Path(preds_path)
    #     preds_path.parent.mkdir(parents=True, exist_ok=True)
    #     print("Shape of preds to be saved:", preds.shape)
    #     np.save(preds_path, preds)
    #     print(f"Saved preds to: {preds_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True, help="Lightning checkpoint path")
    p.add_argument("--config_name", required=True, help="Config name")
    p.add_argument("--root", required=True, help="Dataset root (folder containing data/, videos/, meta/)")
    p.add_argument("--episode", type=int, required=True, help="Episode index to evaluate")
    p.add_argument(
        "--n-step-actions",
        type=int,
        default=None,
        help="Number of action steps to predict at each model call",
    )
    p.add_argument("--save_preds", type=str, default="", help="Optional path to save preds numpy array")
    p.add_argument(
        "--instruction",
        type=str,
        default="Insert the ladder clip (Silver Metal Piece) into the slot with the yellow highlight.",
        help="Natural-language task instruction (SmolVLA).",
    )
    p.add_argument("--override_fps", type=int, default=None, help="Downsample episode to this FPS")

    args = p.parse_args()

    offline_rollout(
        ckpt_path=args.ckpt_path,
        config_name=args.config_name,
        root=Path(args.root),
        episode_index=args.episode,
        save_preds=(args.save_preds or None),
        instruction=args.instruction,
        n_step_actions=args.n_step_actions,
        override_fps=args.override_fps,
    )


if __name__ == "__main__":
    main()
