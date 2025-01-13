from torchmetrics.retrieval import RetrievalRecall, RetrievalNormalizedDCG
from datasets.enums import TestingValue
from scipy.sparse import csr_matrix
import torch
import numpy as np

# TODO: write custom retrival recall

class Eval:
    # Define the metrics
    igr = TestingValue.HOLDOUT.value
    _r_20 = RetrievalRecall(top_k=20, ignore_index=igr)
    _r_50 = RetrievalRecall(top_k=50, ignore_index=igr)
    _r_1000 = RetrievalRecall(top_k=1000, ignore_index=igr)
    _ndcg_100 = RetrievalNormalizedDCG(top_k=100, ignore_index=igr)
    metrics = {
        'Recall20': _r_20,
        'Recall50': _r_50,
        'Recall1000': _r_1000,
        'NDCG10': _ndcg_100
    }
    
    def __init__(self, model, testing_interaction_matrix, device):
        self.model = model
        self.device = device
        # Filter holdout data from testing data by TestingValue.HOLDOUT
        holdout_mask = (testing_interaction_matrix.data == TestingValue.HOLDOUT.value)
        holdout_data = testing_interaction_matrix.copy()
        holdout_data.data = holdout_data.data * holdout_mask
        self.holdout_data = holdout_data
        self.target = torch.tensor(testing_interaction_matrix.toarray()).to(device)
        self.indexes = torch.tensor(np.array([np.full(self.target.shape[1], i) for i in range(self.target.shape[0])]), device=self.device)
        
        
    def __call__(self):
        # get the predictions
        predictions = self.model.predict(self.holdout_data, batch_size=1024)
        # compute the metrics
        return {name: metric(predictions, self.target, self.indexes).item() for name, metric in self.metrics.items()}