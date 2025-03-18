from typing import Protocol
import torch
import numpy as np

class BaseGroupRecommender(Protocol):
    def recommend_for_group(self, group_input_interactions: torch.Tensor, group_target_interactions: torch.Tensor, k: int | None, mask: torch.Tensor | None=None) -> np.ndarray:
        ...