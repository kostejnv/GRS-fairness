from abc import ABC, abstractmethod
import torch
from enum import Enum
from torch.nn import functional as F
import numpy as np

class CombineFeaturesStrategyType(Enum):
    """
    Enum for different types of combine features strategies.
    """
    NONE = "none"
    PERCENTILE = "percentile"
    TOPK = "topk"
    ALL = "all"
    
class CombineFeaturesStrategy(ABC):
    def __init__(self, decoder: torch.Tensor, **kwargs):
        """
        Initialize the CombineFeaturesStrategy with a decoder and any additional keyword arguments.
        
        Args:
            decoder (torch.nn.Module): The decoder module to be used for combining features.
            **kwargs: Additional keyword arguments for specific strategies.
        """
        norm_W = F.normalize(decoder, p=2, dim=1)
        self.sim_matrix = norm_W @ norm_W.T
        self.sim_matrix = self.sim_matrix.fill_diagonal_(0)
        print(self.sim_matrix.shape)
    
    
    @staticmethod
    def get_combine_features_strategy(name: CombineFeaturesStrategyType, **kwargs) -> "CombineFeaturesStrategy":
        if name == CombineFeaturesStrategyType.NONE:
            return NoneCombineFeaturesStrategy(**kwargs)
        elif name == CombineFeaturesStrategyType.PERCENTILE:
            return PercentileCombineFeaturesStrategy(**kwargs)
        elif name == CombineFeaturesStrategyType.TOPK:
            return TopKCombineFeaturesStrategy(**kwargs)
        elif name == CombineFeaturesStrategyType.ALL:
            return AllCombineFeaturesStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown combine features strategy: {name}")
        
    @abstractmethod
    def combine(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        ...
        
        
class NoneCombineFeaturesStrategy(CombineFeaturesStrategy):
    """
    Strategy that does not combine features.
    """
    def __init__(self, decoder: torch.Tensor, **kwargs):
        super().__init__(decoder, **kwargs)
    
    def combine(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        return user_embeddings
    
class PercentileCombineFeaturesStrategy(CombineFeaturesStrategy):
    """
    Strategy that combines features based on a percentile threshold.
    """
    def __init__(self, decoder: torch.Tensor, percentile: float = 0.99, **kwargs):
        super().__init__(decoder, **kwargs)
        sim_values = self.sim_matrix.flatten()
        indices = np.random.randint(0, len(sim_values), (min(100_000, len(sim_values)),))
        sim_values = sim_values[indices]
        self.threshold = torch.quantile(sim_values, percentile)
        self.sim_matrix = torch.where(self.sim_matrix > self.threshold, self.sim_matrix, torch.zeros_like(self.sim_matrix))
        self.sim_matrix = self.sim_matrix.fill_diagonal_(1)
        self.sim_matrix = self.sim_matrix / self.sim_matrix.sum(dim=1, keepdim=True)
        
    def combine(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        return user_embeddings @ self.sim_matrix
    
class TopKCombineFeaturesStrategy(CombineFeaturesStrategy):
    """
    Strategy that combines features based on the top K most similar features.
    """
    def __init__(self, decoder: torch.Tensor, k: int = 5, **kwargs):
        super().__init__(decoder, **kwargs)
        topk = torch.topk(self.sim_matrix, k, dim=1)
        self.sim_matrix = torch.zeros_like(self.sim_matrix)
        self.sim_matrix.scatter_(1, topk.indices, topk.values)
        self.sim_matrix = self.sim_matrix.fill_diagonal_(1)
        self.sim_matrix = self.sim_matrix / self.sim_matrix.sum(dim=1, keepdim=True)
        
    def combine(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        return user_embeddings @ self.sim_matrix
    
class AllCombineFeaturesStrategy(CombineFeaturesStrategy):
    """
    Strategy that combines all features.
    """
    def __init__(self, decoder: torch.Tensor, **kwargs):
        super().__init__(decoder, **kwargs)
        self.sim_matrix = self.sim_matrix.fill_diagonal_(1)
        self.sim_matrix = self.sim_matrix / self.sim_matrix.sum(dim=1, keepdim=True)
        
    def combine(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        return user_embeddings @ self.sim_matrix