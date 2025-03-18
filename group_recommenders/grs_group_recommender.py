from .base_group_recommender import BaseGroupRecommender
from .group_aggregators import AggregationStrategy
from models import ELSA
import torch
import numpy as np
import pandas as pd

class GRSGroupRecommender(BaseGroupRecommender):
    def __init__(self, elsa: ELSA, agregator: AggregationStrategy):
        self.elsa = elsa
        self.agregator = agregator
        
    def recommend_for_group(self, group_input_interactions: torch.Tensor, group_target_interactions: torch.Tensor, k: int | None, mask: torch.Tensor | None=None) -> np.ndarray:
        scores, idxs = self.elsa.recommend(group_input_interactions, None, mask = mask)
        if k is None:
            k = scores.shape[-1]
        # create dataframe user, item, predicted_rating
        data = {'user': [], 'item': [], 'predicted_rating': []}
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                data['user'].append(i)
                data['item'].append(idxs[i, j])
                data['predicted_rating'].append(scores[i, j])
        df = pd.DataFrame(data)
        recommendations = self.agregator.generate_group_recommendations_for_group(df, k)
        return np.array(recommendations[self.agregator.name])