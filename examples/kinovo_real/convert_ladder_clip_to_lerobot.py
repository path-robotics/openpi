"""
Conversion script for converting Path Robotics' LadderClip dataset to LeRobot format.
The dataset was collected using Kinovo's data-saving lerobot format. 

Usage:
uv run examples/kinovo_real/convert_ladder_clip_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/kinovo_real/convert_ladder_clip_to_lerobot.py --data_dir /path/to/your/data --push_to_hub
"""

import shutil
import argparse
from typing import Dict, List

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset


def make_env_state_indices(predict_one_arm: bool, ee_pose_control: bool) -> List[int]:
    """
    Make Indices for environment_state slicing from the low_dim_obs.
    Accounts for whether to use end-effector pose control or not.
    """
    if not ee_pose_control:
        keep = list(range(0, 8)) if predict_one_arm else list(range(0, 8)) + list(range(21, 29))
    else:
        keep = list(range(0, 8)) if predict_one_arm else list(range(0, 8)) + list(range(22, 30))
    return keep

class SliceState(Dataset):
    def __init__(
        self,
        base: MultiLeRobotDataset,
        src="observation.low_dim_obs",
        dst="observation.state",
        keep_idx: List[int] | None = None,
        cast_action_key: str | None = None,
    ):
        self.base = base
        self.src = src
        self.dst = dst
        self.keep_idx = keep_idx
        self.cast_action_key = cast_action_key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        low = s[self.src].to(torch.float32)
        if self.keep_idx is None:
            raise ValueError("keep_idx must be specified")
        else:
            if len(low.shape) == 1:
                env_state = low[self.keep_idx]
            else:
                env_state = low[0, self.keep_idx]
        s[self.dst] = env_state  # (16,)
        if self.cast_action_key and self.cast_action_key in s:
            s[self.cast_action_key] = s[self.cast_action_key].to(torch.float32)
        return s


class RenameKeys(Dataset):
    """Map dataset camera keys and action keys → SmolVLA config keys (string copy)."""

    def __init__(
        self,
        base: Dataset,
        mapping: Dict[str, str],
        src_action_key: str | None = None,
        dst_action_key: str | None = None,
    ):
        self.base, self.mapping = base, mapping
        self.src_action_key = src_action_key
        self.dst_action_key = dst_action_key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        for src, tgt in self.mapping.items():
            if src in s:
                s[tgt] = s[src]
                del s[src]

        if self.src_action_key and self.dst_action_key and self.src_action_key in s:
            s[self.dst_action_key] = s[self.src_action_key]
            del s[self.src_action_key]

        return s


class AddInstruction(Dataset):
    def __init__(self, base: Dataset, text: str):
        self.base, self.text = base, text

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        s["task"] = self.text
        return s
    

def create_dataset():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", 
        default=None,
        required=True,
        help="Parent directory if local; dataset lives at <root>/<repo_id>."
    )
    ap.add_argument(
        "--repo-ids",
        nargs="+",
        required=True,
        help="One or more repo IDs (e.g. setA setB) for training",
    )
    ap.add_argument(
        "--instruction",
        type=str,
        default="Insert the ladder clip (Silver Metal Piece) into the slot with the yellow highlight.",
        help="Natural-language instruction for PI0.",
    )
    args = ap.parse_args()

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image  

    # Meta data for the dataset
    meta0 = LeRobotDatasetMetadata("", root=args.root + args.repo_ids[0])  # any repo_id will do, but using the first one

    state_key = "observation.state"
    low_key = "observation.low_dim_obs"
    action_key = "actions"

    # Map episode for each dataset from all repo_ids
    episodes_map = {}
    for rid in args.repo_ids:
        meta_data = LeRobotDatasetMetadata(rid, root=args.root + rid)
        episodes_map[rid] = list(meta_data.episodes.keys())

    base_train_multi = MultiLeRobotDataset(
        repo_ids=args.repo_ids,
        root=args.root,
        episodes=episodes_map,
    )

    keep_idx = make_env_state_indices(predict_one_arm=False, ee_pose_control=False)

    # Process dataset to match LeRobot format
    train_ds = SliceState(
            base_train_multi, src=low_key, dst=state_key, keep_idx=keep_idx, cast_action_key=action_key
        )
    
    train_ds = AddInstruction(train_ds, text=args.instruction)
    
    return train_ds

if __name__ == "__main__":
    train_ds = create_dataset()
    sample = train_ds[0]
    print("Sample data keys:", sample.keys())
    print("RGB Third View Image Shape:", sample["observation.images.third_view_rgb"].shape)
    