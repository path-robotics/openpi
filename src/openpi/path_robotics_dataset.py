from typing import Any, Dict, Optional, Tuple

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset
from pathlib import Path
import random

SEED = 42


class RobotLearningDatasetMetadata(LeRobotDatasetMetadata):
    def __init__(self, 
        repo_id: str,
        root: str | Path | None = None,
        **kwargs
    ):
        super().__init__(repo_id=repo_id, root=root, **kwargs)

    def _update_splits(self, seed: int = SEED, valid_ratio: float = 0.2, episode_indices: list[int] | None = None) -> None:
        """
        Updates `self.info["splits"]` with episode indices.

        Args:
            seed (int): Seed for reproducibility.
            valid_ratio (float): Fraction of episodes to assign to the validation set.
            episode_indices (list[int]): List of episode indices that are actually available.
        """
        if episode_indices is not None:
            # Use the provided available episodes
            all_indices = episode_indices.copy()
        else:
            # Fall back to total episodes from metadata
            total_episodes = self.info.get("total_episodes", 0)
            if total_episodes == 0:
                return  # No episodes to split
            all_indices = list(range(total_episodes))

        random.seed(seed)
        random.shuffle(all_indices)

        # Ensure splits are always overwritten
        if "splits" in self.info:
            del self.info["splits"]

        valid_size = max(0, min(int(valid_ratio * len(all_indices)), len(all_indices) - 1))
        # Assign indices to train and valid splits
        self.info["splits"] = {
            "train": all_indices[valid_size:],
            "valid": all_indices[:valid_size]
        }

class RobotLearningDataset(LeRobotDataset):
    def __init__(
        self, 
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        video_backend: str = "pyav",
        percent=0.1, 
        mode="train", 
        valid_ratio : float = 0.2,
        **kwargs
    ):
        
         # First, load the dataset to get episode information
        temp_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        temp_episodes = list(temp_metadata.episodes.keys())
        total_episodes = len(temp_episodes)
        # Determine which episodes will actually be used
        if episodes is not None:
            episode_indices = episodes
        else:
            # If no specific episodes provided, use all episodes
            episode_indices = temp_episodes


        dataset_meta = RobotLearningDatasetMetadata(repo_id=repo_id, root=root)
        dataset_meta._update_splits(valid_ratio=valid_ratio, episode_indices=episode_indices)
        dataset_splits = dataset_meta.info["splits"]
        train_indices = dataset_splits["train"]
        self.sampled_indices = None

        video_backend = video_backend

        action_horizon = 1
        if 'action_horizon' in kwargs:
            action_horizon = kwargs.get('action_horizon')
            kwargs.pop('action_horizon', None)
         
        delta_timestamps = None
        if action_horizon > 1:
            delta_timestamps = {
                # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
                "actions": [t / dataset_meta.fps for t in range(action_horizon)],
            }
        
        ## do norm or not -> and pop kwargs to pass in original LeRobotDataset
        self.normalize = kwargs.get('normalize', False)
        kwargs.pop('normalize', None)

        if mode == "train":
            train_indices = dataset_splits["train"]
            super().__init__(repo_id=repo_id, root=root, episodes=train_indices, video_backend=video_backend, delta_timestamps=delta_timestamps, **kwargs)

        elif mode == "valid":
            assert "valid" in dataset_splits, (
                "Validation split not found in dataset_splits. "
                f"Please include a 'valid' key by updating your dataset metadata in {dataset_meta.root}.info.json ."
            )
            valid_indices = dataset_splits["valid"]
            super().__init__(repo_id=repo_id, root=root, episodes=valid_indices, video_backend=video_backend, delta_timestamps=delta_timestamps, **kwargs)

        elif mode == "sample" or episodes is not None:
            super().__init__(repo_id=repo_id, root=root, episodes=episodes, video_backend=video_backend, delta_timestamps=delta_timestamps, **kwargs)
            
        elif mode == "percent" and percent is not None:
            assert 0 < percent <= 1, "Percent should be a value between 0 and 1."

            # Use train indices for percent mode
            train_indices = dataset_splits["train"]
            # Load full dataset first
            super().__init__(repo_id=repo_id, root=root, episodes=train_indices, video_backend=video_backend, delta_timestamps=delta_timestamps, **kwargs)

            # Sample a percentage of frames
            total_frames = len(self)
            num_sampled_frames = int(percent * total_frames)
            self.sampled_indices = sorted(random.sample(range(total_frames), num_sampled_frames))

        else:
            super().__init__(repo_id=repo_id, root=root, video_backend=video_backend, delta_timestamps=delta_timestamps, **kwargs)

        print(f"Actual number of episodes: {total_episodes}")
        print(f"Train indices: {dataset_splits['train']}")
        print(f"Valid indices: {dataset_splits['valid']}")

    def __len__(self):
        """Return the total number of sampled frames if in 'percent' mode, otherwise the full dataset size."""
        if self.sampled_indices is not None:
            return len(self.sampled_indices)
        return super().__len__()

    def __getitem__(self, idx):
        """Fetch frames based on sampled indices in 'percent' mode, otherwise default to full dataset."""
        if self.sampled_indices is not None:
            idx = self.sampled_indices[idx]  # Map index to sampled frames
        batch = super().__getitem__(idx)

        ## normalize ##
        if self.normalize:
            # normalize obs
            obs_max = self.stats['observation.low_dim_obs']['max']
            obs_min = self.stats['observation.low_dim_obs']['min']
            batch['observation.low_dim_obs'] = 2*((batch['observation.low_dim_obs'] - obs_min)/(obs_max - obs_min + 1e-8)) - 1
            batch['observation.low_dim_obs'] = batch['observation.low_dim_obs'].to(torch.float32)

            # normalize actions
            actions_max = self.stats['actions']['max']
            actions_min = self.stats['actions']['min']
            batch['actions'] = 2*((batch['actions'] - actions_min)/(actions_max - actions_min + 1e-8)) - 1  
            batch['actions'] = batch['actions'].to(torch.float32)

            ## TODO: add specific image norm based on imagenet if you want - by default image are norm b/w [0,1]

        ## access only relevant joint pos dims
        ## TODO: remove this by changining config - remove torque and placeholder values from lerobot.yaml
        # new_obs = torch.cat([batch['observation.low_dim_obs'][:8], batch['observation.low_dim_obs'][21:29]], dim=0)
        # batch['observation.low_dim_obs'] = new_obs
        return batch
        

class MultiRobotLearningDataset(MultiLeRobotDataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """
    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        video_backend: str = "pyav",
        **kwargs
    ):
        super().__init__(repo_ids=repo_ids, root=root, episodes=episodes, video_backend=video_backend, **kwargs)



if __name__ == "__main__":
    repo_id = ""
    root = "/home/path/Desktop/robot-learning/data/screw_insertion_1"


    meta = RobotLearningDatasetMetadata(repo_id="", root=root)
    episodes = list(meta.episodes.keys())
    dataset = RobotLearningDataset(repo_id, root=root, episodes=episodes)

    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")
    print(dataset.meta)
    print(dataset.hf_dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=32,
        shuffle=True,
    )
    camera_key = dataset.meta.camera_keys[0]


    for batch in dataloader:
        print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
        print(f"{batch['observation.low_dim_obs'].shape=}")  # (32, 6, c)
        print(f"{batch['actions'].shape=}")  # (32, 64, c)
        break