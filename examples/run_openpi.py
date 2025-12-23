from openpi.training import config as _config
from openpi.policies import policy_config
import openpi.training.data_loader as _data_loader

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata)

config = _config.get_config("pi0_ladderclip_finetune")
checkpoint_dir = "/home/path/Desktop/openpi/checkpoints/pi0_ladderclip_finetune/10000"
print("Checkpoint dir:", checkpoint_dir)


# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print("Policy created:", policy)


# Load LeRobot dataset
data_dir = "/home/path/Desktop/robot-learning/data/ladder_clip_18"
metadata = meta0 = LeRobotDatasetMetadata("", root=data_dir)  # any repo_id will do
dataset = LeRobotDataset(repo_id="",root=data_dir)

sample = dataset[0]

sample['actions'] = sample['actions'][None, :]  # Add batch dimension
sample['observation.low_dim_obs'] = sample['observation.low_dim_obs'][None, :]  # Add batch dimension
sample['observation.images.third_view_rgb'] = sample['observation.images.third_view_rgb'][None, ...]  # Add batch dimension
sample['observation.images.wrist_1_rgb'] = sample['observation.images.wrist_1_rgb'][None, ...]  # Add batch dimension
sample['observation.images.wrist_2_rgb'] = sample['observation.images.wrist_2_rgb'][None, ...]  # Add batch dimension

# Run the policy on a sample.
action = policy.infer(sample)
print("Inferred action Shape:", action['actions'].shape)
