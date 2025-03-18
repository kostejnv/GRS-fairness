import torch
import numpy as np
import random
import scipy.sparse as sp
from datasets import DataLoader
from models import ELSA, SAE, ELSAWithSAE
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
    def split_input_target_interactions(user_item_csr: sp.csr_matrix, target_ratio: float, seed: int = 42) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        np.random.seed(seed)
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
    def _recall_at_k_batch(batch_topk_indices: torch.Tensor, batch_target: torch.Tensor, k: int) -> torch.Tensor:
        target = batch_target.bool()
        predicted_batch = torch.zeros_like(target).scatter(1, batch_topk_indices, torch.ones_like(batch_topk_indices, dtype=bool))
        # recall formula from https://arxiv.org/pdf/1802.05814
        r = (predicted_batch & target).sum(axis=1) / torch.minimum(target.sum(axis=1), torch.ones_like(target.sum(axis=1)) * k)
        return r
    
    @staticmethod
    # implementation from: https://github.com/matospiso/Disentangling-user-embeddings-using-SAE
    def evaluate_recall_at_k_from_elsa(model: ELSA, inputs: DataLoader, targets: DataLoader, k: int) -> np.ndarray:
        recall = []
        for input_batch, target_batch in zip(inputs, targets):
            _, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
            recall.append(Utils._recall_at_k_batch(topk_indices, target_batch, k))
        return torch.cat(recall).detach().cpu().numpy()
    
    @staticmethod
    def evaluate_recall_at_k_from_top_indices(top_indices: np.ndarray, target_batch: torch.Tensor, k: int | None) -> np.ndarray:
        if k is None:
            k = top_indices.shape[-1]
        return Utils._recall_at_k_batch(top_indices, target_batch, k).detach().cpu().numpy()
    
    @staticmethod
    def ndcg_at_k(topk_batch: torch.Tensor, target_batch: torch.Tensor, k: int) -> torch.Tensor:
        target_batch = target_batch.bool()
        relevance = target_batch.gather(1, topk_batch).float()
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
        return dcg / idcg

    @staticmethod
    # implementation from: https://github.com/matospiso/Disentangling-user-embeddings-using-SAE
    def evaluate_ndcg_at_k_from_elsa(model: ELSA, inputs: DataLoader, targets: DataLoader, k: int) -> np.ndarray:
        ndcg = []
        for input_batch, target_batch in zip(inputs, targets):
            _, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
            ndcg.append(Utils.ndcg_at_k(topk_indices, target_batch, k))
        return torch.cat(ndcg).detach().cpu().numpy()
    
    @staticmethod
    def evaluate_ndcg_at_k_from_top_indices(top_indices: np.ndarray, target_batch: torch.Tensor, k: int | None) -> np.ndarray:
        ndcg = []
        if k is None:
            k = top_indices.shape[-1]
        return Utils.ndcg_at_k(top_indices, target_batch, k).detach().cpu().numpy()
    
    @staticmethod
    def evaluate_dense_encoder(model: ELSA, split_csr: sp.csr_matrix, target_ratio: float, batch_size: int, device, seed: int = 42) -> dict[str, float]:
        inputs, targets = Utils.split_input_target_interactions(split_csr, target_ratio, seed)
        inputs = DataLoader(inputs, batch_size, device, shuffle=False)
        targets = DataLoader(targets, batch_size, device, shuffle=False)
        recalls = Utils.evaluate_recall_at_k_from_elsa(model, inputs, targets, k=20)
        ndcgs = Utils.evaluate_ndcg_at_k_from_elsa(model, inputs, targets, k=20)
        return {
            'R20': float(np.mean(recalls)),
            'NDCG20': float(np.mean(ndcgs))
        }
        
    @staticmethod
    def evaluate_sparse_encoder(base_model:ELSA, sae_model:SAE, split_csr: sp.csr_matrix, target_ratio: float, batch_size: int, device, seed: int = 42) -> dict[str, float]:
        inputs, targets = Utils.split_input_target_interactions(split_csr, target_ratio, seed)
        inputs = DataLoader(inputs, batch_size, device, shuffle=False)
        targets = DataLoader(targets, batch_size, device, shuffle=False)
        full = DataLoader(split_csr, batch_size, device, shuffle=False)
        
        fused_model = ELSAWithSAE(base_model, sae_model)
        
        base_model.eval()
        sae_model.eval()
        fused_model.eval()
        
        input_embeddings = np.vstack([base_model.encode(batch).detach().cpu().numpy() for batch in full])
        input_embeddings = DataLoader(input_embeddings, batch_size, device, shuffle=False)
        
        cosines = Utils().evaluate_cosine_similarity(sae_model, input_embeddings)
        l0s = Utils().evaluate_l0(sae_model, input_embeddings)
        dead_neurons = Utils().evaluate_dead_neurons(sae_model, input_embeddings)
        recalls = Utils().evaluate_recall_at_k_from_elsa(base_model, inputs, targets, k=20)
        recalls_with_sae = Utils().evaluate_recall_at_k_from_elsa(fused_model, inputs, targets, k=20)
        recall_degradations = recalls_with_sae - recalls
        ndcgs = Utils().evaluate_ndcg_at_k_from_elsa(base_model, inputs, targets, k=20)
        ndcgs_with_sae = Utils().evaluate_ndcg_at_k_from_elsa(fused_model, inputs, targets, k=20)
        ndcg_degradations = ndcgs_with_sae - ndcgs
        
        return {
            'CosineSim': float(np.mean(cosines)),
            'L0': float(np.mean(l0s)),
            'DeadNeurons': dead_neurons / sae_model.encoder_w.shape[1],
            'R20': float(np.mean(recalls_with_sae)),
            'R20_Degradation': float(np.mean(recall_degradations)),
            'NDCG20': float(np.mean(ndcgs_with_sae)),
            'NDCG20_Degradation': float(np.mean(ndcg_degradations))
        }
        
        
        
    @staticmethod
    def evaluate_cosine_similarity(model, inputs: DataLoader) -> np.ndarray:
        cosine = []
        for input_batch in inputs:
            output_batch = model(input_batch)[0]
            cosine.append(torch.nn.functional.cosine_similarity(input_batch, output_batch, 1))
        return torch.cat(cosine).detach().cpu().numpy()


    @staticmethod
    def evaluate_l0(model, inputs: DataLoader) -> np.ndarray:
        l0s = []
        for input_batch in inputs:
            e = model.encode(input_batch)[0]
            l0s.append((e > 0).float().sum(-1))
        return torch.cat(l0s).detach().cpu().numpy()

    @staticmethod
    def evaluate_dead_neurons(model, inputs: DataLoader) -> int:
        dead_neurons = None
        for input_batch in inputs:
            e = model.encode(input_batch)[0]
            if dead_neurons is None:
                dead_neurons = np.arange(input_batch.shape[1])
            dead_neurons = np.intersect1d(dead_neurons, np.where((e != 0).sum(0).detach().cpu().numpy() == 0)[0])
        return len(dead_neurons)
        
        
    @staticmethod
    def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str) -> None:
        checkpoint = {
            "model_state_dict": model.to('cpu').state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        if '/' in filepath:
            os.makedirs("/".join(filepath.split("/")[:-1]), exist_ok=True)
        torch.save(checkpoint, filepath)
        
    @staticmethod
    def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str, device) -> None:
        checkpoint = torch.load(filepath, weights_only=True, map_location=str(device))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])