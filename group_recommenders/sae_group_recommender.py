from .base_group_recommender import BaseGroupRecommender
from .fusion_strategy import FusionStrategy
from .combine_features_strategy import CombineFeaturesStrategy
from models import ELSA, SAE
import torch
import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SaeGroupRecommender(BaseGroupRecommender):
    def __init__(self, elsa: ELSA, sae: SAE, fusion_strategy: FusionStrategy, combine_features_strategy: CombineFeaturesStrategy):
        self.elsa = elsa
        self.sae = sae
        self.fusion_strategy = fusion_strategy
        self.combine_features_strategy = combine_features_strategy
        
    def recommend_for_group(self, group_input_interactions: torch.Tensor, k: Optional[int], mask: torch.Tensor) -> np.ndarray:          
        group_mask = mask[0]
        dense_embedding = self.elsa.encode(group_input_interactions)
        sparse_embedding, _, x_mean, x_std, _ = self.sae.encode(dense_embedding)
        sparse_embedding = self.combine_features_strategy.combine(sparse_embedding)
        sparse_group_embedding = self.fusion_strategy.normalized_fuse(sparse_embedding)
        dense_group_embedding = self.sae.decode(sparse_group_embedding, torch.mean(x_mean, dim=0), torch.mean(x_std, dim=0))
        scores = self.elsa.decode(dense_group_embedding) - group_mask.float()
        scores = torch.where(group_mask, 0, scores)
        if k is None:
            k = scores.shape[-1]
            
        topscores, idxs = torch.topk(scores, k)
        return np.array(idxs.cpu().numpy())
        
        