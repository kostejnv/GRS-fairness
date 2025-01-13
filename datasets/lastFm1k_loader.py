from datasets.dataset_loader import DatasetLoader
import polars as pl

class LastFm1kLoader(DatasetLoader):
    MIN_USER_INTERACTIONS: int = 20
    MIN_ITEM_INTERACTIONS: int = 20
    
    def __init__(self, path: str = './data/LastFm1k.tsv'):
        super().__init__(path, 'LastFM1k')
        
    def _load(self, path: str) -> None:
        skiprows = [
        2120260-1, 2446318-1, 11141081-1,
        11152099-1, 11152402-1, 11882087-1,
        12902539-1, 12935044-1, 17589539-1
        ]
        
        self.df_interactions = (
            pl.scan_csv(path, separator='\t', has_header=False, quote_char=None)
            .rename({'column_1': 'userId', 'column_3': 'itemId'})
            .select(['userId', 'itemId'])
            .with_row_index().filter(~pl.col("index").is_in(skiprows)).drop("index") # skip damaged rows
            .cast({'userId': pl.String, 'itemId': pl.String})
            .unique()
            .collect()
        )