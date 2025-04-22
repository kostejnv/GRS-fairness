from typing import Protocol
import torch
import numpy as np
from typing import Optional

class BaseGroupRecommender(Protocol):
    def recommend_for_group(self, group_input_interactions: torch.Tensor, k: Optional[int], mask: Optional[torch.Tensor]) -> np.ndarray:
        ...