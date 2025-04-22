import torch
from models import ELSA
from models.sae import SAE
from typing import Optional

class ELSAWithSAE(ELSA):
    def __init__(self, elsa: ELSA, sae: SAE):
        super().__init__(input_dim=elsa.encoder.shape[0], embedding_dim=elsa.encoder.shape[1])
        self.encoder = elsa.encoder
        self.sae = sae
        
    def forward(self, x: torch.Tensor):
        return torch.nn.ReLU()(self.decode(self.sae(self.encode(x))[0]) - x)
    
    @torch.no_grad()
    def recommend(self, interaction_batch: torch.Tensor, k: Optional[int], mask_interactions: bool = True, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.decode(self.sae(self.encode(interaction_batch))[0]) - interaction_batch
        scores = self._normalize_relevance_scores(scores)
        if k is None:
            k = scores.shape[-1]
        if mask_interactions:
            if mask is None:
                mask = interaction_batch != 0
            scores = torch.where(mask, 0, scores)
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()
