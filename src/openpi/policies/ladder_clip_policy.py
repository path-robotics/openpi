# examples/ladderclip/ladderclip_io.py

import dataclasses
from typing import Any, Dict

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model
                 

def _parse_image(image: Any) -> np.ndarray:
    """
    OpenPI expects uint8 HWC images.
    LeRobot samples may come as:
      - numpy uint8 HWC
      - numpy/torch float (0..1) or (0..255)
      - CHW tensors/arrays
      - (1,C,H,W) if time dim exists
    """

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    # drop time dim if present (T,C,H,W) with T==1
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    # CHW -> HWC
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")

    # float -> uint8
    if np.issubdtype(image.dtype, np.floating):
        # assume either 0..1 or 0..255
        if image.max() <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        # ints: just clip/cast to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


@dataclasses.dataclass(frozen=True)
class LadderClipInputs(transforms.DataTransformFn):
    """
    Convert a LadderClip LeRobot sample dict -> OpenPI model inputs.
    Used for both training and inference.
    """

    # Determines which model will be used. Keep as-is.
    model_type: _model.ModelType

    # Indices to keep from low_dim_obs for state
    keep_idx: tuple[int, ...] = tuple(list(range(0, 8)) + list(range(21, 29)))

    # ---- Dataset key mapping (edit if your keys differ) ----
    # Images (from your metadata keys)
    third_view_key: str = "observation.images.third_view_rgb"
    wrist1_key: str = "observation.images.wrist_1_rgb"
    wrist2_key: str = "observation.images.wrist_2_rgb"

    # State: choose ONE depending on your pipeline:
    state_key: str = "observation.low_dim_obs"

    # If you ever renamed actions, update here
    actions_key: str = "actions"

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        base_image = _parse_image(data[self.third_view_key])
        left_wrist = _parse_image(data[self.wrist1_key])
        right_wrist = _parse_image(data[self.wrist2_key])

        low = np.asarray(data[self.state_key], dtype=np.float32).reshape(-1)
        state = low[list(self.keep_idx)]  


        inputs: Dict[str, Any] = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # For PI0_FAST, they keep mask True even for padding; keep their rule.
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.True_,
            },
        }

        # Actions exist during training
        if self.actions_key in data:
            inputs["actions"] = np.asarray(data[self.actions_key], dtype=np.float32)

        # Prompt / instruction
        inputs["prompt"] = "Insert the ladder clip (Silver Metal Piece) into the slot with the yellow highlight."

        return inputs


@dataclasses.dataclass(frozen=True)
class LadderClipOutputs(transforms.DataTransformFn):
    """
    Convert OpenPI model outputs -> LadderClip action format (inference only).
    """

    action_dim: int  # set this to your dataset action dimension (e.g., 16)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Model outputs actions as (T, A_model); return only first action_dim
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
