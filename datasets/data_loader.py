import torch
import numpy as np
import scipy.sparse as sp

class DataLoader:
    def __init__(self, data, batch_size: int, device: torch.device, shuffle: bool = False):
        self.data = data
        self.dataset_size = self.data.shape[0]
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self.dataset_size // self.batch_size + (self.dataset_size % self.batch_size != 0)

    def __iter__(self):
        self.permutation = np.random.permutation(self.dataset_size) if self.shuffle else np.arange(self.dataset_size)
        self.i = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self.i >= self.dataset_size:
            raise StopIteration
        next_i = min(self.i + self.batch_size, self.dataset_size)
        batch = self.data[self.permutation[self.i : next_i]]
        self.i = next_i
        return torch.tensor(batch.toarray() if isinstance(batch, sp.csr_matrix) else batch, device=self.device)