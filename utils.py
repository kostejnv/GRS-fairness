import torch
import numpy as np
import random
import scipy.sparse as sp
from datasets import DataLoader
from models import ELSA, SAE, ELSAWithSAE
import os
from typing import Optional

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
    def rel_ndcg_at_k(batch_topk_indices: np.ndarray, batch_rel_scores: np.ndarray, k: int) -> np.ndarray:
        """
        Compute nDCG@k for a batch of predictions.

        Parameters:
        - batch_topk_indices: (batch_size, k) array of indices of top-k predicted items.
        - batch_rel_scores: (batch_size, num_items) array of relevance scores.
        - k: cutoff rank.

        Returns:
        - ndcg_scores: (batch_size,) array of nDCG scores.
        """
        # Get the actual relevance scores for the predicted top-k indices
        batch_size = batch_topk_indices.shape[0]
        topk_rels = np.take_along_axis(batch_rel_scores, batch_topk_indices, axis=1)

        # Compute DCG@k
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = (topk_rels * discounts).sum(axis=1)

        # Compute IDCG@k by sorting the true relevance scores
        sorted_rels = np.sort(batch_rel_scores, axis=1)[:, ::-1][:, :k]
        idcg = (sorted_rels * discounts).sum(axis=1)

        # To avoid division by zero
        idcg[idcg == 0.0] = 1e-10

        # Compute nDCG@k
        ndcg = dcg / idcg

        return ndcg
        
    
    @staticmethod
    # implementation from: https://github.com/matospiso/Disentangling-user-embeddings-using-SAE
    def evaluate_recall_at_k_from_elsa(model: ELSA, inputs: DataLoader, targets: DataLoader, k: int) -> np.ndarray:
        recall = []
        for input_batch, target_batch in zip(inputs, targets):
            _, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
            recall.append(Utils._recall_at_k_batch(torch.tensor(topk_indices).to(target_batch.device), target_batch, k))
        return torch.cat(recall).detach().cpu().numpy()
    
    @staticmethod
    def evaluate_recall_at_k_from_top_indices(top_indices: np.ndarray, target_batch: torch.Tensor, k: Optional[int]) -> np.ndarray:
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
            ndcg.append(Utils.ndcg_at_k(torch.tensor(topk_indices).to(target_batch.device), target_batch, k))
        return torch.cat(ndcg).detach().cpu().numpy()
    
    @staticmethod
    def evaluate_ndcg_at_k_from_top_indices(top_indices: np.ndarray, target_batch: torch.Tensor, k: Optional[int]) -> np.ndarray:
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
        
        
        
        embeddings = np.vstack([base_model.encode(batch).detach().cpu().numpy() for batch in full])
        embeddings_dataloader = DataLoader(embeddings, batch_size, device, shuffle=False)
        
        embeddings = torch.tensor(embeddings, device=device)
        reconstructed_embeddings = torch.tensor(np.vstack([sae_model(batch)[0].detach().cpu().numpy() for batch in embeddings_dataloader]), device=device)
        sparse_embeddings = torch.tensor(np.vstack([sae_model.encode(batch)[0].detach().cpu().numpy() for batch in embeddings_dataloader]), device=device)
        
        elsa_recommendations = np.vstack([base_model.recommend(batch, 20, mask_interactions=True)[1] for batch in inputs])
        elsa_rec_dataloader = DataLoader(elsa_recommendations, batch_size, device, shuffle=False)
        sae_recommendations = np.vstack([fused_model.recommend(batch, 20, mask_interactions=True)[1] for batch in inputs])
        sae_rec_dataloader = DataLoader(sae_recommendations, batch_size, device, shuffle=False)
        
        cosines = Utils().evaluate_cosine_similarity(embeddings, reconstructed_embeddings)
        l0s = Utils().evaluate_l0(sparse_embeddings)
        dead_neurons = Utils().evaluate_dead_neurons(sparse_embeddings)
        recalls = np.mean(np.concatenate([Utils()._recall_at_k_batch(recs, targs, 20).cpu().numpy() for recs, targs in zip(elsa_rec_dataloader, targets)]))
        recalls_with_sae = np.mean(np.concatenate([Utils()._recall_at_k_batch(recs, targs, 20).cpu().numpy() for recs, targs in zip(sae_rec_dataloader, targets)]))
        recall_degradations = recalls_with_sae - recalls
        ndcgs = np.mean(np.concatenate([Utils().ndcg_at_k(recs, targs, 20).cpu().numpy() for recs, targs in zip(elsa_rec_dataloader, targets)]))
        ndcgs_with_sae = np.mean(np.concatenate([Utils().ndcg_at_k(recs, targs, 20).cpu().numpy() for recs, targs in zip(sae_rec_dataloader, targets)]))
        ndcg_degradations = ndcgs_with_sae - ndcgs

        
        return {
            'CosineSim': float(np.mean(cosines)),
            'L0': float(np.mean(l0s)),
            'DeadNeurons': dead_neurons / sae_model.encoder_w.shape[1],
            'R20': float(recalls_with_sae),
            'R20_Degradation': float(recall_degradations),
            'NDCG20_base': float(ndcgs),
            'NDCG20': float(ndcgs_with_sae),
            'NDCG20_Degradation': float(ndcg_degradations)
        }
        
        
        
    @staticmethod
    def evaluate_cosine_similarity(embeddings: torch.Tensor, reconstructed_embeddings: torch.Tensor) -> np.ndarray:
        return torch.nn.functional.cosine_similarity(embeddings, reconstructed_embeddings, dim=1).detach().cpu().numpy()


    @staticmethod
    def evaluate_l0(sparse_embeddings: torch.Tensor) -> np.ndarray:
        return (sparse_embeddings > 0).float().sum(-1).detach().cpu().numpy()

    @staticmethod
    def evaluate_dead_neurons(sparse_embeddings: torch.Tensor) -> int:
        return int((sparse_embeddings.sum(0) == 0).sum().detach().cpu().numpy())
    
    @staticmethod
    def rel_score_per_item(recommendations: np.ndarray, elsa_scores: np.ndarray) -> np.ndarray:
        return np.array([elsa_scores[:, rec] for rec in recommendations])
    
    @staticmethod
    def ranks_per_item(recommendations: np.ndarray, elsa_scores: np.ndarray) -> np.ndarray:
        # reverse argsort
        ranks = np.argsort(-elsa_scores, axis=1)
        ranks = np.argsort(ranks, axis=1)
        ranks = 1 - ( np.log10(ranks + 1) / np.log10(elsa_scores.shape[1]))
        ranks = np.array([ranks[:, rec] for rec in recommendations])
        return ranks
    
    @staticmethod
    def recommendations_similarity(recommendations: np.ndarray, elsa: ELSA):
        # Extract item embeddings for the recommended items
        item_embeddings = elsa.encoder[recommendations]
        
        # Compute pairwise cosine similarity
        pairwise_cosine_sim = torch.nn.functional.cosine_similarity(
            item_embeddings.unsqueeze(1), item_embeddings.unsqueeze(0), dim=-1
        )
        
        # Exclude self-similarity by setting diagonal to 0
        pairwise_cosine_sim.fill_diagonal_(0)
        
        # Exclude diagonal from mean calculation
        num_elements = pairwise_cosine_sim.numel() - pairwise_cosine_sim.shape[0]
        mean_cosine_similarity = pairwise_cosine_sim.sum().item() / num_elements
        
        return float(mean_cosine_similarity)
        
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