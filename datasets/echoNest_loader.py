from datasets.dataset_loader import DatasetLoader
import polars as pl

class EchoNestLoader(DatasetLoader):
    MIN_USER_INTERACTIONS: int = 50
    MIN_ITEM_INTERACTIONS: int = 500
    def __init__(self, path: str = './data/EchoNest.txt'):
        super().__init__(path, 'EchoNest')
        
    def _load(self, path: str) -> None:
        self.df_interactions = (
            pl.scan_csv(path, separator='\t', has_header=False)
            .rename({'column_1': 'userId', 'column_2': 'itemId'})
            .select(['userId', 'itemId'])
            .cast({'userId': pl.String, 'itemId': pl.String})
            .unique()
            .collect()
        )