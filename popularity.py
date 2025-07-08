from scipy.sparse import csr_matrix
import numpy as np

class Popularity:
    def __init__(self, crs_interactions: csr_matrix):
        self.crs_interactions = crs_interactions
        self.popularity = None
        self._calculate_popularity()
        
    def _calculate_popularity(self):
        frequncies = self.crs_interactions.sum(axis=0).A1
        self.popularity = frequncies /frequncies.max()
        
    def popularity_score(self, ids: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: self.popularity[x])(ids)