from typing import Protocol, List, Tuple
import scipy.sparse as sp
import numpy as np
from abc import abstractmethod
import polars as pl
import torch
import time
import logging
import os

class DatasetLoader():
    # Constants
    MIN_USER_INTERACTIONS: int = 20
    MIN_ITEM_INTERACTIONS: int = 200
    # -------------------
    # Instance variables
    df_interactions: pl.DataFrame
    csr_interactions: sp.csr_matrix
    users: np.ndarray
    items: np.ndarray
    train_csr: sp.csr_matrix
    valid_csr: sp.csr_matrix
    test_csr: sp.csr_matrix
    train_users: np.ndarray
    valid_users: np.ndarray
    test_users: np.ndarray
    
    def __init__(self, path: str, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.path = path
        
        
    def prepare(self, cfg) -> None:
        '''Prepare all data for processing'''
        if not os.path.exists(self.path):
            raise FileNotFoundError(f'Dataset file not found: {self.path}. Download the dataset and place it in the correct path')
        
        self.logger.info('Preparing dataset: %s', self.name)
        start = time.time()
        
        cached_file = '.'.join(self.path.split('.')[:-1]) + '.arrow'
        if os.path.exists(cached_file): # load cached dataset
            self.logger.info(f'The dataset has already been prepared. Loading cached file...')
            self.df_interactions = pl.read_ipc(cached_file)
            filter_end = time.time()
            self.logger.info(f'Dataset loaded in {filter_end - start:.2f}s')
        else: # load dataset from scratch
            self.logger.info(f'Loading dataset...')
            self._load(self.path)
            load_end = time.time()
            self.logger.info(f'Dataset loaded in {load_end - start:.2f}s')
            
            # check that df_interactions is correctly loaded
            if not isinstance(self.df_interactions, pl.DataFrame) or any(col not in self.df_interactions.columns for col in ['userId', 'itemId']):
                raise ValueError('df_interactions must be a pl.DataFrame with columns userId and itemId')
            
            self.logger.info(f'Filtering dataset...')
            self._filter()
            filter_end = time.time()
            self.logger.info(f'Dataset filtered in {filter_end - load_end:.2f}s')
            
            # save df_interactions for faster loading
            self.df_interactions.write_ipc(cached_file)
            self.logger.info(f'Dataset saved in {cached_file}')
            
            
        self.logger.info("Final interactions: %d, users: %d, items: %d", len(self.df_interactions), len(self.df_interactions['userId'].unique()), len(self.df_interactions['itemId'].unique()))
        
        self.logger.info(f'Creating csr_matrix...')
        self._create_csr_matrix()
        csr_end = time.time()
        self.logger.info(f'csr_matrix created in {csr_end - filter_end:.2f}s')
        
        self.logger.info(f'Splitting dataset...')
        self._split(cfg.val_ratio, cfg.test_ratio, seed=cfg.seed)
        split_end = time.time()
        self.logger.info(f'Dataset split in {split_end - csr_end:.2f}s')
        
        self.logger.info('-'*20)
        self.logger.info(f'Dataset prepared in {split_end - start:.2f}s')
        
    @abstractmethod
    def _load(self, path: str) -> None:
        ''' Abstract that loads a pl.DataFrame with 2 columns - userId, itemId - into variable self.df_interactions'''
        ...
        
    def _filter(self) -> None:
        '''Filter users and items that has too few interactions'''        
        df = self.df_interactions
        self.logger.info('Initial interactions: %d, users: %d, items: %d', len(df), len(df['userId'].unique()), len(df['itemId'].unique()))
        
        # convert columns to categorical
        df = df.cast({"userId": pl.String, "itemId": pl.String}).cast({"userId": pl.Categorical, "itemId": pl.Categorical})
        
        # filter users with too few interactions
        df = df.filter(
            df['userId'].is_in(df['userId'].value_counts().filter(pl.col('count') >= self.MIN_USER_INTERACTIONS)['userId']),
        )
        # filter items with too few interactions
        df = df.filter(
            df['itemId'].is_in(df['itemId'].value_counts().filter(pl.col('count') >= self.MIN_ITEM_INTERACTIONS)['itemId']),
        )
        
        # reset categories
        df = df.cast({"userId": pl.String, "itemId": pl.String}).cast({"userId": pl.Categorical, "itemId": pl.Categorical})
        
        self.logger.info('Filtered interactions: %d, users: %d, items: %d', len(df), len(df['userId'].unique()), len(df['itemId'].unique()))
        
        self.df_interactions = df
        
    def _create_csr_matrix(self) -> None:
        '''Create a csr_matrix from the interactions DataFrame'''
        self.users = self.df_interactions['userId'].cat.get_categories().to_numpy()
        self.items = self.df_interactions['itemId'].cat.get_categories().to_numpy()

        self.csr_interactions = sp.csr_matrix(
            (np.ones(len(self.df_interactions), dtype=np.float32), (
                self.df_interactions['userId'].to_physical().to_numpy(),
                self.df_interactions['itemId'].to_physical().to_numpy()
            )),
            shape=(len(self.users), len(self.items))
        )
        
    def _split(self, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> None:
        '''Split the dataset into train, validation and test sets'''
        np.random.seed(seed)
        p = np.random.permutation(len(self.users))
        val_count, test_count = int(len(self.users) * val_ratio), int(len(self.users) * test_ratio)
        train_idx, val_idx, test_idx = p[val_count+test_count:], p[:val_count], p[val_count:val_count+test_count]

        self.train_users, self.train_csr = self.users[train_idx], self.csr_interactions[train_idx]
        self.valid_users, self.valid_csr = self.users[val_idx], self.csr_interactions[val_idx]
        self.test_users, self.test_csr = self.users[test_idx], self.csr_interactions[test_idx]