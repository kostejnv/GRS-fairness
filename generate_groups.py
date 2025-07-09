# IMPORTS

import mlflow
from copy import deepcopy
import torch
from datasets import LastFm1kLoader, DataLoader, MovieLensLoader
from utils import Utils
from models import ELSA
import tqdm
import numpy as np
from torch.nn import functional as F
import random
import os
import plotly.express as px
import pandas as pd
import logging
import argparse
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_arguments():
    parser = argparse.ArgumentParser() # TODO: Add description
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # dataset
    parser.add_argument('--dataset', type=str, default='MovieLens', help='Dataset to use. For now, only "LastFM1k" and "MovieLens" are supported')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--td_quantile', type=float, default=0.75, help='Threshold quantile for similarity')
    parser.add_argument('--ts_quantile', type=float, default=0.25, help='Threshold quantile for divergence')
    parser.add_argument('--similar_groups', type=list, default=[3], help='Number of similar groups')
    parser.add_argument('--divergent_groups', type=list, default=[], help='Number of divergent groups')
    parser.add_argument('--random_groups', type=list, default=[3], help='Number of random groups')
    parser.add_argument('--opposing_groups', type=list, default=[[2,1]], help='Number of opposing groups')
    parser.add_argument('--user_sample', type=int, default=20_000, help='Number of users to sample')
    parser.add_argument('--group_count', type=int, default=100_000, help='Number of groups')
    parser.add_argument('--final_group_count', type=int, default=1_000, help='Final number of group after all filtering')
    parser.add_argument('--common_interactions', type=int, default=5, help='Common interaction count that the final groups should have')
    parser.add_argument("--run_id", type=str, default='0c2c7c4b7cd5427db21b9c7022ffbc18', help="Mlflow Run ID of the base model")
    parser.add_argument("--out_dir", type=str, default='data/synthetic_groups', help="Output directory")
    parser.add_argument("--user_set", type=str, default='test', help="User set to generate groups for (full, test, valid, train)")
    
    return parser.parse_args()

def main(args):
    Utils.set_seed(args.seed)
    device = Utils.set_device()
    
    run = mlflow.get_run(args.run_id)
    artifact_path = run.info.artifact_uri
    artifact_path = './' + artifact_path[artifact_path.find('mlruns'):]
    
    params = run.data.params
    
    assert params['model'] == 'ELSA', 'Model from run is not ELSA -> not supported'
    assert params['dataset'] == args.dataset, 'Dataset from run is not the same as the current dataset -> not supported'
    
    # load model

    items = int(params['items'])
    factors = int(params['factors'])

    model = ELSA(items, factors).to(device)
    optimizer = torch.optim.Adam(model.parameters()) # not used, but needed for loading
    Utils.load_checkpoint(model, optimizer, f'{artifact_path}/checkpoint.ckpt', device)
    model.eval()
    logging.info('Model loaded')
    
    if args.dataset == 'LastFM1k':
        dataset_loader = LastFm1kLoader()
    elif args.dataset == 'MovieLens':
        dataset_loader = MovieLensLoader()
    else:
        raise ValueError(f'Dataset {args.dataset} not supported. Check typos.')
    dataset_loader.prepare(args)
    
    logging.info('Dataset loaded')
    
    # load interactions
    if args.user_set == 'full':
        user_sample = min(args.user_sample, dataset_loader.csr_interactions.shape[0])
        user_idx = np.random.permutation(dataset_loader.csr_interactions.shape[0])[:user_sample]
    elif args.user_set == 'train':
        user_sample = min(args.user_sample, len(dataset_loader.train_idx))
        user_idx = np.random.choice(dataset_loader.train_idx, user_sample, replace=False)
    elif args.user_set == 'valid':
        user_sample = min(args.user_sample, len(dataset_loader.valid_idx))
        user_idx = np.random.choice(dataset_loader.valid_idx, user_sample, replace=False)
    elif args.user_set == 'test':
        user_sample = min(args.user_sample, len(dataset_loader.test_idx))
        user_idx = np.random.choice(dataset_loader.test_idx, user_sample, replace=False)
    else:
        raise ValueError(f'User set {args.user_set} not supported. Check typo.')
    
    csr_interactions = dataset_loader.csr_interactions[user_idx]
    user_ids = dataset_loader.users[user_idx]
    
    logging.info(f'Interactions shape: {csr_interactions.shape}')

    interactions_batches = DataLoader(csr_interactions, batch_size=1024, device=device, shuffle=False)

    # create user embeddings
    batches_embeddings = []
    for batch in tqdm.tqdm(interactions_batches, desc='Creating user embeddings'):
        batch_embeddings = model.encode(batch)
        batches_embeddings.append(batch_embeddings.detach())
    user_embeddings = torch.cat(batches_embeddings)
    
    # compute similarity matrix
    normalized_embeddings = F.normalize(user_embeddings, p=2, dim=1).to(torch.float16)
    similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
    similarity_matrix = (similarity_matrix + 1) / 2 # opposite user embeddings are completely opposite, so we normalize to [0,1]
    logging.info(f'Similarity matrix shape: {similarity_matrix.shape}')

    
    # compute thresholds
    
    # select 100 000 values to compute quantiles without flattening the matrix
    indices = np.random.randint(0, len(similarity_matrix), (min(100_000, len(similarity_matrix)**2), 2))
    logging.info(f'Sample {len(indices)} values to compute quantiles')
    similarity_values = similarity_matrix[indices[:,0], indices[:,1]]
    # filter out all 1 values
    similarity_values = similarity_values[similarity_values != 1].to(torch.float32)
    
    logging.info(f'Similarity values shape: {similarity_values.shape}')

    Td = similarity_values.quantile(args.td_quantile)
    Ts = similarity_values.quantile(args.ts_quantile)

    logging.info(f'Td: {Td}, Ts: {Ts}')
    
    def is_user_similar(group_members, user):
        if not group_members:
            return True
        mean_similarity = similarity_matrix[group_members, user].mean()
        return mean_similarity >= Td

    def _similar_group(group_size):
        user_count = len(similarity_matrix)
        group_members = [random.randint(0, user_count-1)]
        rest = list(set(range(user_count)) - set(group_members))
        for _ in range(1_000): # it can be blocked in a loop if the group does not exists
            user = random.choice(rest)
            if is_user_similar(group_members, user):
                group_members.append(user)
                rest.remove(user)
            if len(group_members) >= group_size:
                return group_members
        raise TimeoutError('Could not find similar group')

    def similar_group(group_size):
        while True:
            try:
                return _similar_group(group_size)
            except TimeoutError:
                logging.info('Could not find similar group... trying again')
                pass

    def is_user_divergent(group_members, user):
        if not group_members:
            return True
        mean_similarity = similarity_matrix[group_members, user].mean()
        return mean_similarity <= Ts

    def _divergent_group(group_size):
        user_count = len(similarity_matrix)
        group_members = [random.randint(0, user_count-1)]
        rest = list(set(range(user_count)) - set(group_members))
        for _ in range(1_000): # it can be blocked in a loop if the group does not exists
            user = random.choice(rest)
            if is_user_divergent(group_members, user):
                group_members.append(user)
                rest.remove(user)
            if len(group_members) >= group_size:
                return group_members
        raise TimeoutError('Could not find divergent group')

    def divergent_group(group_size):
        while True:
            try:
                return _divergent_group(group_size)
            except TimeoutError:
                logging.info('Could not find divergent group... trying again')
                pass
            

    def _opposing_group(group_size):
        user_count = len(similarity_matrix)
        groups_lefts = deepcopy(group_size)
        
        # choose the first user
        group_members = ([random.randint(0, user_count-1)], [])
        groups_lefts[0] -= 1
        
        rest = list(set(range(user_count)) - set(group_members[0]))
        
        index = 1
        for _ in range(1_000):
            # choose the subgroup to expand
            group_to_expand = index % 2
            if groups_lefts[group_to_expand] == 0:
                index += 1
                group_to_expand = index % 2
                
            user = random.choice(rest)
            if is_user_similar(group_members[group_to_expand], user) and is_user_divergent(group_members[1-group_to_expand], user):
                group_members[group_to_expand].append(user)
                groups_lefts[group_to_expand] -= 1
                rest.remove(user)
                
            if groups_lefts[0] == 0 and groups_lefts[1] == 0:
                return group_members[0] + group_members[1]
            index += 1
        raise TimeoutError('Could not find opposing group')

    def opposing_group(group_size):
        while True:
            try:
                return _opposing_group(group_size)
            except TimeoutError:
                logging.info('Could not find opposing group... trying again')
                pass
            
    def random_group(group_size):
        user_count = len(similarity_matrix)
        group_members = [random.randint(0, user_count-1) for _ in range(group_size)]
        return group_members
            
            
    logging.info('Generating groups')
    similar_groups = {}
    for group_size in args.similar_groups:
        group_idxs = []
        for i in range(args.group_count):
            print(f"Generating similar group {i+1}/{args.group_count}")
            group_idxs.append(similar_group(group_size))
        similar_groups[group_size] = user_ids[group_idxs]
    logging.info('Similar groups generated')
    
    divergent_groups = {}
    for group_size in args.divergent_groups:
        group_idxs = []
        for i in range(args.group_count):
            print(f"Generating divergent group {i+1}/{args.group_count}")
            group_idxs.append(divergent_group(group_size))
        divergent_groups[group_size] = user_ids[group_idxs]
    logging.info('Divergent groups generated')
    
    opposing_groups = {}
    for group_size in args.opposing_groups:
        group_idxs = []
        for i in range(args.group_count):
            print(f"Generating opposing group {i+1}/{args.group_count}")
            group_idxs.append(opposing_group(group_size))
        opposing_groups[tuple(group_size)] = user_ids[group_idxs]
    logging.info('Opposing groups generated')
    
    random_groups = {}
    for group_size in args.random_groups:
        group_idxs = []
        for i in range(args.group_count):
            print(f"Generating random group {i+1}/{args.group_count}")
            group_idxs.append(random_group(group_size))
        random_groups[group_size] = user_ids[group_idxs]
    logging.info('Random groups generated')
        
    def filter_groups(original_groups, group_size):
        # filter groups with same users
        original_groups = [list(group) for group in original_groups if len(set(group)) == len(group)]
        # filter same groups
        original_groups = list(set([tuple(group) for group in original_groups]))
        original_groups = np.array(original_groups)
        groups = np.vectorize(lambda x: np.argwhere(dataset_loader.users == x))(original_groups)
        print(f"Number of groups: {len(groups)}")
            
        # get common interactions
        common_items = []
        for group in groups:
            group_interactions = dataset_loader.csr_interactions[group]
            _, targets = Utils.split_input_target_interactions_for_groups(group_interactions, 0.5)
            ci = (targets.sum(axis=0) == group_size).astype(int).sum()
            common_items.append(ci)
            
        positive_groups = (np.array(common_items) > args.common_interactions)
        np.random.seed(args.seed)
        selected_groups = original_groups[positive_groups]
        print(f"Groups fulfiling all conditions: {len(selected_groups)}")
        original_groups = np.random.permutation(selected_groups)[:args.final_group_count]
        return original_groups
    
    # save groups as numpy arrays

    out_path = f'{args.out_dir}/{args.dataset}/{args.user_set}'
    os.makedirs(out_path, exist_ok=True)

    for group_size, groups in similar_groups.items():
        if groups.size == 0:
            continue
        print(f"Filtering similar groups of size {group_size}")
        groups = filter_groups(groups, group_size)
        np.save(f'{out_path}/similar_{group_size}_{args.user_set}.npy', np.array(groups))
        
    for group_size, groups in divergent_groups.items():
        if groups.size == 0:
            continue
        print(f"Filtering divergent groups of size {group_size}")
        groups = filter_groups(groups, group_size)
        np.save(f'{out_path}/divergent_{group_size}_{args.user_set}.npy', np.array(groups))
        
    for group_size, groups in opposing_groups.items():
        if groups.size == 0:
            continue
        print(f"Filtering opposing groups of size {group_size}")
        groups = filter_groups(groups, sum(group_size))
        np.save(f'{out_path}/opposing_{group_size[0]}_{group_size[1]}_{args.user_set}.npy', np.array(groups))

    for group_size, groups in random_groups.items():
        if groups.size == 0:
            continue
        print(f"Filtering random groups of size {group_size}")
        groups = filter_groups(groups, group_size)
        np.save(f'{out_path}/random_{group_size}_{args.user_set}.npy', np.array(groups))
        
    logging.info('Groups saved')
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)