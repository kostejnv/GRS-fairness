# IMPORTS

import mlflow
from copy import deepcopy
import torch
from datasets import EchoNestLoader, LastFm1kLoader, DataLoader
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
    parser.add_argument('--dataset', type=str, default='LastFM1k', help='Dataset to use. For now, only "LastFM1k" and "EchoNest" are supported')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--td_quantile', type=float, default=0.9, help='Threshold quantile for similarity')
    parser.add_argument('--ts_quantile', type=float, default=0.1, help='Threshold quantile for divergence')
    parser.add_argument('--similar_groups', type=list, default=[3,5], help='Number of similar groups')
    parser.add_argument('--divergent_groups', type=list, default=[3,5], help='Number of divergent groups')
    parser.add_argument('--opposing_groups', type=list, default=[[2,1],[3,2],[4,1]], help='Number of opposing groups')
    parser.add_argument('--group_count', type=int, default=75, help='Number of groups')
    parser.add_argument("--run_id", type=str, default='34ade4833e9e48d9b2d3c504a0af4346', help="Run ID of the base model")
    parser.add_argument("--out_dir", type=str, default='data/synthetic_groups', help="Output directory")
    parser.add_argument("--user_set", type=str, default='full', help="User set to generate groups for (full, test)")
    
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
    Utils.load_checkpoint(model, optimizer, f'{artifact_path}/checkpoint.ckpt')
    model.eval()
    logging.info('Model loaded')
    
    if args.dataset == 'EchoNest':
        dataset_loader = EchoNestLoader()
    elif args.dataset == 'LastFM1k':
        dataset_loader = LastFm1kLoader()
    else:
        raise ValueError(f'Dataset {args.dataset} not supported. Check typos.')
    dataset_loader.prepare(args)
    
    logging.info('Dataset loaded')
    
    # load interactions
    if args.user_set == 'full':
        csr_interactions = dataset_loader.csr_interactions
    elif args.user_set == 'test':
        csr_interactions = dataset_loader.test_csr
    else:
        raise ValueError(f'User set {args.user_set} not supported. Check typo.')

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
    indices = np.random.randint(0, len(similarity_matrix), (100_000, 2))
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
                pass
            
            
    logging.info('Generating groups')
    similar_groups = {}
    for group_size in args.similar_groups:
        similar_groups[group_size] = [similar_group(group_size) for _ in range(args.group_count)]
    logging.info('Similar groups generated')
    
    divergent_groups = {}
    for group_size in args.divergent_groups:
        divergent_groups[group_size] = [divergent_group(group_size) for _ in range(args.group_count)]
    logging.info('Divergent groups generated')
    
    opposing_groups = {}
    for group_size in args.opposing_groups:
        opposing_groups[str(group_size)] = [opposing_group(group_size) for _ in range(args.group_count)]
    logging.info('Opposing groups generated')
    
    # save groups as numpy arrays

    out_path = f'{args.out_dir}/{args.dataset}/{args.user_set}'
    os.makedirs(out_path, exist_ok=True)

    for group_size, groups in similar_groups.items():
        np.save(f'{out_path}/similar_{group_size}.npy', np.array(groups))
        
    for group_size, groups in divergent_groups.items():
        np.save(f'{out_path}/divergent_{group_size}.npy', np.array(groups))
        
    for group_size, groups in opposing_groups.items():
        np.save(f'{out_path}/opposing_{group_size[1]}_{group_size[3]}.npy', np.array(groups))
        
    logging.info('Groups saved')
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)