import torch
import numpy as np
import random
import scipy.sparse as sp
from datasets import DataLoader
from models import ELSA
import os

class Utils:
    @staticmethod
    def set_seed(seed: int) -> None:
        torch.manual_seed(seed)  # CPU seed
        torch.mps.manual_seed(seed)  # Metal seed
        torch.cuda.manual_seed(seed)  # GPU seed
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # NumPy seed
        random.seed(seed)  # Python seed
        
    @staticmethod
    def set_device():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device('mps') if torch.backends.mps.is_available() else device
        return device
    
    @staticmethod
    def split_input_target_interactions(user_item_csr: sp.csr_matrix, target_ratio: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        target_mask = np.concatenate(
            [
                np.random.permutation(np.array([True] * int(np.ceil(row_nnz * target_ratio)) + [False] * int((row_nnz - np.ceil(row_nnz * target_ratio)))))
                for row_nnz in np.diff(user_item_csr.indptr)
            ]
        )
        inputs: sp.csr_matrix = user_item_csr.copy()
        targets: sp.csr_matrix = user_item_csr.copy()
        
        inputs.data *= ~target_mask
        targets.data *= target_mask
        inputs.eliminate_zeros()
        targets.eliminate_zeros()
        return inputs, targets

    @staticmethod
    # implementation from: https://github.com/matospiso/Disentangling-user-embeddings-using-SAE
    def evaluate_recall_at_k(model: ELSA, inputs: DataLoader, targets: DataLoader, k: int) -> np.ndarray:
        recall = []
        for input_batch, target_batch in zip(inputs, targets):
            _, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
            topk_indices = torch.tensor(topk_indices, device=target_batch.device)
            target_batch = target_batch.bool()
            predicted_batch = torch.zeros_like(target_batch).scatter_(1, topk_indices, torch.ones_like(topk_indices, dtype=bool))
            # recall formula from https://arxiv.org/pdf/1802.05814
            r = (predicted_batch & target_batch).sum(axis=1) / torch.minimum(target_batch.sum(axis=1), torch.ones_like(target_batch.sum(axis=1)) * k)
            recall.append(r)
        return torch.cat(recall).detach().cpu().numpy()

    @staticmethod
    # implementation from: https://github.com/matospiso/Disentangling-user-embeddings-using-SAE
    def evaluate_ndcg_at_k(model: ELSA, inputs: DataLoader, targets: DataLoader, k: int) -> np.ndarray:
        ndcg = []
        for input_batch, target_batch in zip(inputs, targets):
            _, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
            topk_indices = torch.tensor(topk_indices, device=target_batch.device)
            target_batch = target_batch.bool()
            relevance = target_batch.gather(1, topk_indices).float()
            # DCG@k
            gains = 2**relevance - 1
            discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
            dcg = (gains / discounts).sum(dim=1)
            # IDCG@k (ideal DCG)
            sorted_relevance, _ = torch.sort(target_batch.float(), dim=1, descending=True)
            ideal_gains = 2 ** sorted_relevance[:, :k] - 1
            ideal_discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
            idcg = (ideal_gains / ideal_discounts).sum(dim=1)
            idcg[idcg == 0] = 1
            # nDCG@k
            ndcg.append(dcg / idcg)
        return torch.cat(ndcg).detach().cpu().numpy()
    
    @staticmethod
    def evaluate(model: ELSA, split_csr: sp.csr_matrix, target_ratio: float, batch_size: int, device) -> dict[str, float]:
        inputs, targets = Utils.split_input_target_interactions(split_csr, target_ratio)
        inputs = DataLoader(inputs, batch_size, device, shuffle=False)
        targets = DataLoader(targets, batch_size, device, shuffle=False)
        recalls = Utils.evaluate_recall_at_k(model, inputs, targets, k=20)
        ndcgs = Utils.evaluate_ndcg_at_k(model, inputs, targets, k=20)
        return {
            'R20': float(np.mean(recalls)),
            'NDCG20': float(np.mean(ndcgs))
        }
        
    @staticmethod
    def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str) -> None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if '/' in filepath:
            os.makedirs("/".join(filepath.split("/")[:-1]), exist_ok=True)
        torch.save(checkpoint, filepath)
        
    @staticmethod
    def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str) -> None:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])