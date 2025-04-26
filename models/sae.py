from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True)


class SAE(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, cfg: dict):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.cfg = cfg
        self.reconstruction_loss = cfg.get("reconstruction_loss", "Cosine")
        self.device = cfg.get("device", "cpu")
        self.n_batches_to_dead = cfg.get("n_batches_to_dead", 5)
        self.topk_aux = cfg.get("topk_aux", 512)
        self.normalize = cfg.get("normalize", False)
        
        
        
        self.encoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([input_dim, embedding_dim])))
        self.encoder_b = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([embedding_dim, input_dim])))
        self.decoder_b = nn.Parameter(torch.zeros(input_dim))
        # memory of inactive neurons
        self.inactive_neurons = torch.zeros(embedding_dim, device=self.device)
        
        # use transpose encoder as decoder to eliminate dead neurons
        self.decoder_w.data = self.encoder_w.t().data
        self.normalize_decoder()

    @abstractmethod
    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def total_loss(self, partial_losses: dict) -> torch.Tensor:
        raise NotImplementedError

    def _compute_loss_dict(self, x: torch.Tensor, e_pre: torch.Tensor, e: torch.Tensor, x_out: torch.Tensor, standardized_x: torch.Tensor, e_positive: Optional[torch.Tensor]) -> dict[str, torch.Tensor]:
        losses = {
            "L2": (x_out - x).pow(2).mean(),
            "L1": e.abs().sum(-1).mean(),
            "L0": (e > 0).float().sum(-1).mean(),
            "Cosine": (1 - F.cosine_similarity(x, x_out, 1)).mean(),
            "Auxiliary": self._auxiliary_loss(standardized_x, e_pre, e),
            "Contrastive": self._contrastive_loss(e, e_positive) if e_positive is not None else torch.zeros(1, device=x.device),
        }
        losses["Loss"] = self.total_loss(losses)
        return losses
    
    def compute_loss_dict(self, batch: torch.Tensor, positive_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        out, e, e_pre, batch_mean, batch_std, standardized_batch = self(batch)
        e_positive = self.encode(positive_batch)[0] if positive_batch is not None else None
        return self._compute_loss_dict(batch, e_pre, e, out, standardized_batch, e_positive)
    
    def _auxiliary_loss(self, x: torch.Tensor, e_pre: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        dead_neurons = self.inactive_neurons > self.n_batches_to_dead
        if dead_neurons.sum() == 0:
            return torch.zeros(1, device=x.device)
        
        e_topk_aux = torch.topk(
            e_pre[:, dead_neurons],
            min(self.topk_aux, dead_neurons.sum()),
            dim=-1,
        )
        
        e_aux = torch.zeros_like(e_pre[:, dead_neurons]).scatter(
            -1,
            e_topk_aux.indices,
            e_topk_aux.values,
        )
        
        return (x - (e @ self.decoder_w + e_aux @ self.decoder_w[dead_neurons] + self.decoder_b)).pow(2).mean()
        
    def _contrastive_loss(self, e: torch.Tensor, e_positive: torch.Tensor) -> torch.Tensor:
        e = F.normalize(e, dim=-1)
        e_positive = F.normalize(e_positive, dim=-1)

        logits_1 = torch.matmul(e, e_positive.T)
        logits_2 = torch.matmul(e_positive, e.T)

        targets = torch.arange(e.shape[0], device=e.device)
        loss_1 = F.cross_entropy(logits_1, targets)
        loss_2 = F.cross_entropy(logits_2, targets)

        return (loss_1 + loss_2) / 2

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, x_mean, x_std = self.standardize_input(x)
        e_pre = F.relu((x - self.decoder_b) @ self.encoder_w + self.encoder_b)
        e = self.post_process_embedding(e_pre)
        if self.normalize:
            e = l2_normalize(e)
        if self.training:
            self._update_inactive_neurons(e)
        return e, e_pre, x_mean, x_std, x

    def decode(self, e: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        return self.destandardize_output(e @ self.decoder_w + self.decoder_b, x_mean, x_std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e, e_pre, x_mean, x_std, standardized_x = self.encode(x)
        out = self.decode(e, x_mean, x_std)
        return out, e, e_pre, x_mean, x_std, standardized_x

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder_w.data = l2_normalize(self.decoder_w.data)
        if self.decoder_w.grad is not None:
            self.decoder_w.grad -= (self.decoder_w.grad * self.decoder_w.data).sum(-1, keepdim=True) * self.decoder_w.data

    def standardize_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x_mean = x.mean(dim=-1, keepdim=True)
        x -= x_mean
        x_std = x.std(dim=-1, keepdim=True) + 1e-7
        x /= x_std
        return x, x_mean, x_std

    def destandardize_output(self, out: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        return x_mean + out * x_std

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor, positive_batch: Optional[torch.Tensor]) -> dict[str, torch.Tensor]:
        self.train()
        losses = self.compute_loss_dict(batch, positive_batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        self.normalize_decoder()
        optimizer.step()
        return losses
    
    def _update_inactive_neurons(self, e: torch.Tensor) -> None:
        self.inactive_neurons += (e.sum(0) == 0).float()
        self.inactive_neurons[e.sum(0) > 0] = 0


class BasicSAE(SAE):
    def __init__(self, input_dim: int, embedding_dim: int, cfg: dict):
        super().__init__(input_dim, embedding_dim, cfg)
        self.reconstruction_coef = cfg.get("reconstruction_coef", 1.0)
        self.auxiliary_coef = cfg.get("auxiliary_coef", 0)
        self.contrastive_coef = cfg.get("contrastive_coef", 0)

    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        return e

    def total_loss(self, partial_losses: dict) -> torch.Tensor:
        reconstruction_loss = partial_losses[self.reconstruction_loss] + self.cfg["l1_coef"] * partial_losses["L1"]
        auxiliary_loss = partial_losses["Auxiliary"]
        return reconstruction_loss + self.cfg["auxiliary_coef"] * auxiliary_loss


class TopKSAE(SAE):
    def __init__(self, input_dim: int, embedding_dim: int, cfg: dict):
        super().__init__(input_dim, embedding_dim, cfg)
        self.k = cfg["k"]
        self.reconstruction_coef = cfg.get("reconstruction_coef", 1.0)
        self.auxiliary_coef = cfg.get("auxiliary_coef", 0)
        self.contrastive_coef = cfg.get("contrastive_coef", 0)
        self.l1_coef = cfg.get("l1_coef", 0)

    def post_process_embedding(self, e: torch.Tensor) -> torch.Tensor:
        e_topk = torch.topk(e, self.cfg["k"], dim=-1)
        return torch.zeros_like(e).scatter(-1, e_topk.indices, e_topk.values)

    def total_loss(self, partial_losses: dict) -> torch.Tensor:
        rec_coef, aux_coef, con_coef = self.cfg["reconstruction_coef"], self.cfg["auxiliary_coef"], self.cfg["contrastive_coef"]
        
        reconstruction_loss = rec_coef * partial_losses[self.reconstruction_loss] + self.l1_coef * partial_losses["L1"]
        auxiliary_loss = aux_coef * partial_losses["Auxiliary"]
        contrastive_loss = con_coef * partial_losses["Contrastive"]
        
        return reconstruction_loss + auxiliary_loss + contrastive_loss
