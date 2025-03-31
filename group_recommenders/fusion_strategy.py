from abc import ABC, abstractmethod
import torch
from enum import Enum

class FusionStrategyType(Enum):
    """
    Enum for different types of fusion strategies.
    """
    AVERAGE = "average"
    

class FusionStrategy(ABC):
    """
    Abstract base class for fusion strategies for user sparse embeddings.
    """
        
    @staticmethod
    def get_fusion_strategy(name: FusionStrategyType) -> "FusionStrategy":
        if name == FusionStrategyType.AVERAGE:
            return AverageFusionStrategy()
        else:
            raise ValueError(f"Unknown fusion strategy: {name}")
        

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