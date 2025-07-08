from abc import ABC, abstractmethod
import torch
from enum import Enum

class FusionStrategyType(Enum):
    """
    Enum for different types of fusion strategies.
    """
    AVERAGE = "average"
    SQUARE_AVERAGE = "square_average"
    TOPK_MEAN = "topk"
    MAX = "max"
    MIN = "min"
    COMMON_FEATURES = "common_features"
    AT_LEAST_2_COMMON_FEATURES = "at_least_2_common_features"
    WCOM = 'wcom'  # Weighted Commonalities
    

class FusionStrategy(ABC):
    """
    Abstract base class for fusion strategies for user sparse embeddings.
    """
        
    @staticmethod
    def get_fusion_strategy(name: FusionStrategyType, **kwargs) -> "FusionStrategy":
        if name == FusionStrategyType.AVERAGE:
            return AverageFusionStrategy()
        elif name == FusionStrategyType.SQUARE_AVERAGE:
            return SquareAverageFusionStrategy()
        elif name == FusionStrategyType.TOPK_MEAN:
            return TopKMeanFusionStrategy(**kwargs)
        elif name == FusionStrategyType.MAX:
            return MaxFusionStrategy()
        elif name == FusionStrategyType.MIN:
            return MinFusionStrategy()
        elif name == FusionStrategyType.COMMON_FEATURES:
            return CommonFeaturesFusionStrategy()
        elif name == FusionStrategyType.AT_LEAST_2_COMMON_FEATURES:
            return AtLeast2CommonFeaturesFusionStrategy()
        elif name == FusionStrategyType.WCOM:
            return WeightedCommonalitiesFusionStrategy()
        else:
            raise ValueError(f"Unknown fusion strategy: {name}")
    
    def normalized_fuse(self, group_members_embeddings: torch.Tensor, normalize_user_embedding: bool = False) -> torch.Tensor:
        """
        Normalize the group embeddings according to mean user embedding sum.
        Args:
            group_members_embeddings (torch.Tensor): A tensor of shape (n_group_members, embedding_dim) representing the embeddings of the group members.
        Returns:
            torch.Tensor: A tensor of shape (embedding_dim,) representing the normalized group embedding.
        """
        mean_member_norm = group_members_embeddings.norm(dim=1).mean()
        if normalize_user_embedding:
            group_members_embeddings = group_members_embeddings / group_members_embeddings.norm(dim=1, keepdim=True)
        group_embedding = self.fuse(group_members_embeddings)
        
        return group_embedding / group_embedding.norm() * mean_member_norm

    @abstractmethod
    def fuse(self, group_members_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse the group members' embeddings into a single group embedding.
        Args:
            group_members_embeddings (torch.Tensor): A tensor of shape (n_group_members, embedding_dim) representing the embeddings of the group members.
        Returns:
            torch.Tensor: A tensor of shape (embedding_dim,) representing the fused group embedding.
        """
        ...
        
class AverageFusionStrategy(FusionStrategy):
    def fuse(self, group_members_embeddings: torch.Tensor) -> torch.Tensor:
        return group_members_embeddings.mean(dim=0)
    
class SquareAverageFusionStrategy(FusionStrategy):
    def fuse(self, group_members_embeddings: torch.Tensor) -> torch.Tensor:
        return group_members_embeddings.mean(dim=0) ** 2
    
class MaxFusionStrategy(FusionStrategy):
    def fuse(self, group_members_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.max(group_members_embeddings, dim=0)[0]
    
class MinFusionStrategy(FusionStrategy):
    def fuse(self, group_members_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.min(group_members_embeddings, dim=0)[0]

class CommonFeaturesFusionStrategy(FusionStrategy):
    def fuse(self, group_members_embeddings: torch.Tensor) -> torch.Tensor:
        binarized = group_members_embeddings > 0
        common_features = binarized.all(dim=0).repeat(group_members_embeddings.shape[0], 1)
        masked_group_embedding = torch.where(common_features, group_members_embeddings, 0)
        return torch.mean(masked_group_embedding, dim=0)
    
class AtLeast2CommonFeaturesFusionStrategy(FusionStrategy):
    def fuse(self, group_members_embeddings: torch.Tensor) -> torch.Tensor:
        binarized = group_members_embeddings > 0
        common_features = (binarized.sum(dim=0) >= 2).repeat(group_members_embeddings.shape[0], 1)
        masked_group_embedding = torch.where(common_features, group_members_embeddings, 0)
        return torch.mean(masked_group_embedding, dim=0)
    
class WeightedCommonalitiesFusionStrategy(FusionStrategy):
    def fuse(self, group_members_embeddings: torch.Tensor) -> torch.Tensor:
        binarized = group_members_embeddings > 0
        commonallities = binarized.sum(dim=0).float()
        return torch.mean(group_members_embeddings, dim=0) * commonallities

class TopKMeanFusionStrategy(FusionStrategy):
    def __init__(self, k: int = 64):
        self.k = k
    
    def fuse(self, group_members_embeddings: torch.Tensor) -> torch.Tensor:
        embedding = group_members_embeddings.mean(dim=0)
        kth_value = torch.topk(embedding, self.k)[0][-1]
        mask = embedding >= kth_value
        return embedding * mask.float()
    
    