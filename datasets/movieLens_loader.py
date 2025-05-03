from datasets.dataset_loader import DatasetLoader
import polars as pl

class MovieLensLoader(DatasetLoader):
    MIN_USER_INTERACTIONS: int = 50
    MIN_ITEM_INTERACTIONS: int = 20
    def __init__(self, path: str = './data/MovieLens.csv'):
        super().__init__(path, 'MovieLens')
        
    def _load(self, path: str) -> None:
        self.df_interactions = (
            pl.scan_csv(path, has_header=True)
            .select(['userId', 'movieId', 'rating'])
            .rename({'movieId': 'itemId'})
            .cast({'userId': pl.String, 'itemId': pl.String, 'rating': pl.Float64})
            .filter(pl.col('rating') >= 4.0)
            .select(['userId', 'itemId'])
            .unique()
            .sort(['userId', 'itemId'])
            .collect()
        )