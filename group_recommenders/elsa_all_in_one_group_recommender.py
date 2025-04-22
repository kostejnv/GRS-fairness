from .base_group_recommender import BaseGroupRecommender
from .fusion_strategy import FusionStrategy
from models import ELSA, SAE
import torch
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ElsaAllInOneGroupRecommender(BaseGroupRecommender):
    def __init__(self, elsa: ELSA):
        self.elsa = elsa
        
    def recommend_for_group(self, group_input_interactions: torch.Tensor, k: int | None, mask: torch.Tensor) -> np.ndarray:
        group_mask = mask[0]
        dense_embedding = self.elsa.encode(group_mask.float())
        scores = torch.nn.ReLU()(self.elsa.decode(dense_embedding) - group_mask.float())
        scores = torch.where(group_mask, 0, scores)
        if k is None:
            k = scores.shape[-1]
            
        topscores, idxs = torch.topk(scores, k)
        return np.array(idxs.cpu().numpy())
        
        