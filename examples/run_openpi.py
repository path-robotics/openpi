import time
from openpi.training import config as _config
from openpi.policies import policy_config

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata)

config = _config.get_config("pi0_ladderclip_finetune")
checkpoint_dir = "/home/path/Desktop/openpi/checkpoints/pi0_ladderclip_finetune/20000"
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
# Warmup (compilation)
_ = policy.infer(sample)

# Timed run
start = time.time()
op = policy.infer(sample)
end = time.time()
print("Inferred action Shape:", op['actions'].shape)
print("Inference time (s):", end - start)