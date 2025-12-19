from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import droid_policy


config = _config.get_config("pi05_droid")
checkpoint_dir = "/home/path/models/openpi/pi05_droid_pytorch"
print("Checkpoint dir:", checkpoint_dir)


# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# print("Policy created:", policy)


# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = droid_policy.make_droid_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape) 