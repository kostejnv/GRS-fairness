import torch
from models import ELSA
from models.sae import SAE

class ELSAWithSAE(ELSA):
    def __init__(self, elsa: ELSA, sae: SAE):
        super().__init__(input_dim=elsa.encoder.shape[0], embedding_dim=elsa.encoder.shape[1])
        self.encoder = elsa.encoder
        self.sae = sae
        
    def forward(self, x: torch.Tensor):
        return torch.nn.ReLU()(self.decode(self.sae(self.encode(x))[0]) - x)