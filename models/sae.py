from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True)


class SAE(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, reconstruction_loss: str):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.encoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([input_dim, embedding_dim])))
        self.encoder_b = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([embedding_dim, input_dim])))
        self.decoder_b = nn.Parameter(torch.zeros(input_dim))
        
        # use transpose encoder as decoder to eliminate dead neurons
        self.decoder_w.data = self.encoder_w.t().data
        self.normalize_decoder()

    @abstractmethod
    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def total_loss(self, partial_losses: dict) -> torch.Tensor:
        raise NotImplementedError

    def _compute_loss_dict(self, x: torch.Tensor, e_pre: torch.Tensor, e: torch.Tensor, x_out: torch.Tensor) -> dict:
        losses = {
            "L2": (x_out - x).pow(2).mean(),
            "L1": e.abs().sum(-1).mean(),
            "L0": (e > 0).float().sum(-1).mean(),
            "Cosine": (1 - F.cosine_similarity(x, x_out, 1)).mean(),
        }
        losses["Loss"] = self.total_loss(losses)
        return losses

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, x_mean, x_std = self.standardize_input(x)
        e_pre = F.relu((x - self.decoder_b) @ self.encoder_w + self.encoder_b)
        return self.post_process_embedding(e_pre), e_pre, x_mean, x_std

    def decode(self, e: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        return self.destandardize_output(e @ self.decoder_w + self.decoder_b, x_mean, x_std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e, e_pre, x_mean, x_std = self.encode(x)
        out = self.decode(e, x_mean, x_std)
        return out, e, e_pre, x_mean, x_std

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder_w.data = l2_normalize(self.decoder_w.data)
        if self.decoder_w.grad is not None:
            self.decoder_w.grad -= (self.decoder_w.grad * self.decoder_w.data).sum(-1, keepdim=True) * self.decoder_w.data

    def standardize_input(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=-1, keepdim=True)
        x -= x_mean
        x_std = x.std(dim=-1, keepdim=True) + 1e-7
        x /= x_std
        return x, x_mean, x_std

    def destandardize_output(self, out: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        return x_mean + out * x_std

    def compute_loss_dict(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        out, e, e_pre, batch_mean, batch_std = self(batch)
        return self._compute_loss_dict(batch, e_pre, e, out)

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = self.compute_loss_dict(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        self.normalize_decoder()
        optimizer.step()
        return losses


class BasicSAE(SAE):
    def __init__(self, input_dim: int, embedding_dim: int, reconstruction_loss: str, **extra_params: dict):
        super().__init__(input_dim, embedding_dim, reconstruction_loss)
        self.l1_coef = extra_params["l1_coef"]

    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        return e

    def total_loss(self, partial_losses: dict) -> torch.Tensor:
        return partial_losses[self.reconstruction_loss] + self.l1_coef * partial_losses["L1"]


class TopKSAE(SAE):
    def __init__(self, input_dim: int, embedding_dim: int, reconstruction_loss: str, **extra_params: dict):
        super().__init__(input_dim, embedding_dim, reconstruction_loss)
        self.l1_coef = extra_params["l1_coef"]
        self.k = extra_params["k"]

    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        e_topk = torch.topk(e, self.k, dim=-1)
        return torch.zeros_like(e).scatter(-1, e_topk.indices, e_topk.values)

    def total_loss(self, partial_losses: dict) -> torch.Tensor:
        return partial_losses[self.reconstruction_loss] + self.l1_coef * partial_losses["L1"]
