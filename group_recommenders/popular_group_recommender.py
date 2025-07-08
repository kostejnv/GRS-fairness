from .base_group_recommender import BaseGroupRecommender
from .fusion_strategy import FusionStrategy
from models import ELSA, SAE
import torch
import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class PopularGroupRecommender(BaseGroupRecommender):
    def __init__(self, crs_matrix):
        self.interactions = crs_matrix
        self.popularity = self.interactions.sum(axis=0).A1
        
    def recommend_for_group(self, group_input_interactions: torch.Tensor, k: Optional[int], mask: torch.Tensor) -> np.ndarray:
        group_mask = mask[0].cpu()
        scores = torch.tensor(self.popularity)
        scores = torch.where(group_mask, 0, scores)

        if k is None:
            k = scores.shape[-1]            
        topscores, idxs = torch.topk(scores, k)
        return np.array(idxs.cpu().numpy())
        
        