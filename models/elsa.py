import torch
import torch.nn as nn
import torch.optim as optim


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True)


def normalized_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return (l2_normalize(y_pred) - l2_normalize(y_true)).pow(2).sum(-1).mean()


class ELSA(nn.Module):
    """Scalable Linear Shallow Autoencoder
    Paper: https://dl.acm.org/doi/abs/10.1145/3523227.3551482"""

    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([input_dim, embedding_dim])))
        self.normalize_encoder()

    @torch.no_grad()
    def normalize_encoder(self) -> None:
        self.encoder.data = l2_normalize(self.encoder.data)
        if self.encoder.grad is not None:
            self.encoder.grad -= (self.encoder.grad * self.encoder.data).sum(-1, keepdim=True) * self.encoder.data

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.encoder

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        return e @ self.encoder.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.ReLU()(self.decode(self.encode(x)) - x)

    def compute_loss_dict(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"Loss": normalized_mse_loss(self(batch), batch)}

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = self.compute_loss_dict(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        self.normalize_encoder()
        optimizer.step()
        return losses

    @torch.no_grad()
    def recommend(self, interaction_batch: torch.Tensor, k: int | None, mask_interactions: bool = True, mask: torch.Tensor|None = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self(interaction_batch)
        if k is None:
            k = scores.shape[-1]
        if mask_interactions:
            if mask is None:
                mask = interaction_batch != 0
            scores = torch.where(mask, 0, scores)
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()
