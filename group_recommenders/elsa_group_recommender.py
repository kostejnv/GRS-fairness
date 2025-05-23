from .base_group_recommender import BaseGroupRecommender
from .fusion_strategy import FusionStrategy
from models import ELSA, SAE
import torch
import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ElsaGroupRecommender(BaseGroupRecommender):
    def __init__(self, elsa: ELSA, fusion_strategy: FusionStrategy):
        self.elsa = elsa
        self.fusion_strategy = fusion_strategy
        
    def recommend_for_group(self, group_input_interactions: torch.Tensor, k: Optional[int], mask: torch.Tensor) -> np.ndarray:
        group_mask = mask[0]
        dense_embedding = self.elsa.encode(group_input_interactions)
        dense_group_embedding = self.fusion_strategy.fuse(dense_embedding)
        scores = self.elsa.decode(dense_group_embedding) - group_mask.float()
        scores = torch.where(group_mask, 0, scores)
        if k is None:
            k = scores.shape[-1]
            
        topscores, idxs = torch.topk(scores, k)
        return np.array(idxs.cpu().numpy())
        
        