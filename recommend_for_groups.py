import argparse
import logging
import sys
import torch
from datasets import EchoNestLoader, LastFm1kLoader, DataLoader
from models import ELSA, ELSAWithSAE, BasicSAE, TopKSAE, SAE
import mlflow
import numpy as np
import random
import os
import datetime
from tqdm import tqdm
from utils import Utils
from copy import deepcopy
from group_recommenders import BaseGroupRecommender
import scipy.sparse as sp
from group_recommenders import AggregationStrategy, GRSGroupRecommender

TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.info(f'Recommending with GR System - {TIMESTAMP}')
device = Utils.set_device()
logging.info(f'Device: {device}')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LastFM1k', help='Dataset to use. For now, only "LastFM1k" and "EchoNest" are supported')
    # model parameters
    parser.add_argument("--sae_run_id", type=str, help="Run ID of the analyzed SAE model")
    parser.add_argument("--use_base_model_from_sae", action='store_true', help="Use base model from SAE run")
    parser.add_argument("--base_run_id", type=str, default='a54d3546cd884e2a99e5792c80844aab', help="Run ID of the base model if not using SAE base model")
    
    # Recommender parameters
    parser.add_argument("--recommender_strategy", type=str, default='ADD', help="Strategy to use for recommending. Options: 'SAE', 'ADD', ...") # TODO: Add more strategies
    parser.add_argument("--SAE_fusion_strategy", type=str, help="Only for SAE strategy. Strategy to fuse user sparse embeddings.") # TODO: Add more strategies
    
    # group parameters
    parser.add_argument("--group_type", type=str, default='sim', help="Type of group to analyze. Options: 'sim', 'div', '21'")
    parser.add_argument("--group_size", type=int, default=3, help="Size of the group to analyze")
    parser.add_argument("--user_set", type=str, default='train', help="User set from which the groups where sampled (full, test, train)")
    # stable parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--target_ratio', type=float, default=0.2, help='Ratio of target interactions')
    parser.add_argument('--k', type=int, default=20, help='Evaluation at k')
    
    return parser.parse_args()

GROUP_TYPES = ['sim', 'div', '21']
RECOMMENDER_STRATEGIES = ['SAE', 'ADD', 'LMS', 'GFAR', 'EPFuzzDA']
SAE_FUSION_STRATEGIES = ['max']
GROUP_SIZES = [3]

def get_groups_path(dataset, group_type, group_size, user_set):
    path = f'data/synthetic_groups/{dataset}/{user_set}/'
    if group_type == 'sim':
        filename = f'similar_{group_size}.npy'
    elif group_type == 'div':
        filename = f'divergent_{group_size}.npy'
    elif group_type == '21':
        filename = f'opposing_2_1.npy'
    return path + filename

def recommend(args, group_recommender: BaseGroupRecommender, elsa: ELSA, groups, interactions: sp.csr_matrix):
    logging.info('Recommending for groups')
    
    mlflow.set_experiment(f'GR_{args.dataset}_{args.group_type}_{args.group_size}')
    mlflow.set_experiment_tags({
        'Dataset': args.dataset,
        'Group Type': args.group_type,
        'Group Size': args.group_size,
        'Task': 'Group Recommending',
        'mlflow.note.content': f'This experiment was created to evaluate group recommending for {args.dataset} dataset using {args.group_type} groups of size {args.group_size}.'
    })
    
    if args.recommender_strategy == 'SAE':
        run_name = f'{args.recommender_strategy}_{args.SAE_fusion_strategy}_{args.top_k}_{args.embedding_dim}'
    else:
        run_name = f'{args.recommender_strategy}'
        
    with mlflow.start_run(run_name=run_name):
        ndcgs, recalls = [], []
        for group in tqdm(groups, desc='Group Recommending', total=len(groups)):
            group_interactions = interactions[group]
            inputs, targets = Utils.split_input_target_interactions(group_interactions, args.target_ratio, args.seed)
            inputs, targets = torch.tensor(inputs.toarray(), device=device), torch.tensor(targets.toarray(), device=device)
            mask = (inputs.sum(axis=0).squeeze() != 0).unsqueeze(0).repeat(inputs.shape[0], 1)
            # targets should be masked with negative mask
            targets = targets * (~mask)
            recommendations = group_recommender.recommend_for_group(inputs, targets, args.k, mask)
            recommendations = torch.tensor(recommendations, device=device).unsqueeze(0).repeat(inputs.shape[0], 1)
            recalls.append(Utils.evaluate_recall_at_k_from_top_indices(recommendations, targets, args.k).mean())
            ndcgs.append(Utils.evaluate_ndcg_at_k_from_top_indices(recommendations, targets, args.k))
        ndcgs_means = [np.mean(ndcgs[i]) for i in range(len(ndcgs))]
        ndcgs_mins = [np.min(ndcgs[i]) for i in range(len(ndcgs))]
        ndcgs_maxs = [np.max(ndcgs[i]) for i in range(len(ndcgs))]
        mlflow.log_metrics({
            f'R{args.k}/mean': float(np.mean(recalls)),
            f'R{args.k}/std': float(np.std(recalls)),
            f'NDCG{args.k}/mean': float(np.mean(ndcgs_means)),
            f'NDCG{args.k}/std': float(np.std(ndcgs_means)),
            f'NDCG{args.k}/min': float(np.mean(ndcgs_mins)),
            f'NDCG{args.k}/max': float(np.mean(ndcgs_maxs)),
            f'NDCG{args.k}/min:max': float(np.mean(ndcgs_mins) / np.mean(ndcgs_maxs)),
        })
        

def main(args):
    assert args.group_size in GROUP_SIZES, 'Only group size 3 is supported for now'
    assert args.group_type in ['sim', 'div', '21'], 'Group type not supported'
    assert args.recommender_strategy in RECOMMENDER_STRATEGIES, 'Recommender strategy not supported'
    if not args.recommender_strategy == 'SAE' or not args.use_base_model_from_sae:
        assert args.base_run_id is not None, 'Base model run ID is required'
    
    # Load dataset
    if args.dataset == 'LastFM1k':
        dataset_loader = LastFm1kLoader()
    elif args.dataset == 'EchoNest':
        dataset_loader = EchoNestLoader()
    dataset_loader.prepare(args)
    interactions = dataset_loader.csr_interactions
    
    # Load groups
    path = get_groups_path(args.dataset, args.group_type, args.group_size, args.user_set)
    if not os.path.exists(path):
        raise FileNotFoundError(f'File {path} not found')
    
    groups = np.load(path, allow_pickle=True)
    groups = np.vectorize(lambda x: np.argwhere(dataset_loader.users == x))(groups) # convert user indexes to user ids
    logging.info(f'Groups with shape {groups.shape} loaded from {path}')
    
    # everythin what is needed for SAE
    if args.recommender_strategy == 'SAE':
        raise NotImplementedError('SAE strategy not implemented yet')
        assert args.SAE_fusion_strategy in SAE_FUSION_STRATEGIES, 'SAE fusion strategy not supported'
        assert args.sae_run_id is not None, 'SAE run ID is required'
    
    
        # load sae model
        sae_run = mlflow.get_run(args.sae_run_id)
        sae_params = sae_run.data.params
        
        assert sae_params['dataset'] == args.dataset, 'SAE model dataset does not match current dataset'
        
        # log parameters
        args.embedding_dim = int(sae_params['embedding_dim'])
        args.top_k = sae_params.get('top_k', None)
        args.sample_users = sae_params['sample_users']
        args.sae_model = sae_params['model']
        args.sae_run_id = args.sae_run_id
    
        sae_artifact_path = sae_run.info.artifact_uri
        sae_artifact_path = './' + sae_artifact_path[sae_artifact_path.find('mlruns'):]
        
        sae_embedding_dim = int(sae_params['embedding_dim'])
        sae_reconstruction_loss = sae_params['reconstruction_loss']
        sae_l1_coef = float(sae_params['l1_coef'])
        sae_top_k = int(sae_params['top_k'])
        base_factors = int(sae_params['base_factors'])
        
        if sae_params['model'] == 'BasicSAE':
            sae = BasicSAE(base_factors, sae_embedding_dim, sae_reconstruction_loss, l1_coef=sae_l1_coef).to(device)
        elif sae_params['model'] == 'TopKSAE':
            sae = TopKSAE(base_factors, sae_embedding_dim, sae_reconstruction_loss, l1_coef=sae_l1_coef, k=sae_top_k).to(device)
        else:
            raise ValueError(f'Model {sae_params["model"]} not supported. Check typos.')
        
        sae_optimizer = torch.optim.Adam(sae.parameters())
        Utils.load_checkpoint(sae, sae_optimizer, f'{sae_artifact_path}/checkpoint.ckpt', device)
        sae.to(device)
        sae.eval()
        logging.info(f'SAE model loaded from {sae_artifact_path}')

        
    # load base model
    assert args.base_run_id or (args.use_base_model_from_sae and args.sae_run_id is not None), 'Base model run ID is required'
    base_model_id = args.base_run_id if not args.use_base_model_from_sae else sae_params['base_run_id']
    base_model_run = mlflow.get_run(base_model_id)
    base_model_params = base_model_run.data.params
    
    base_artifact_path = base_model_run.info.artifact_uri
    base_artifact_path = './' + base_artifact_path[base_artifact_path.find('mlruns'):]
    
    # Load models
    base_factors = int(base_model_params['factors'])
    base_items = int(base_model_params['items'])
    
    elsa = ELSA(base_items, base_factors)
    optimizer = torch.optim.Adam(elsa.parameters())
    Utils.load_checkpoint(elsa, optimizer, f'{base_artifact_path}/checkpoint.ckpt', device)
    elsa.to(device)
    elsa.eval()
    logging.info(f'ELSA model loaded from {base_artifact_path}')

    if args.recommender_strategy == 'SAE':
        group_recommender = ... # TODO: Implement SAE group recommender
    else: # other strategies
        aggregator = AggregationStrategy.getAggregator(args.recommender_strategy)
        group_recommender = GRSGroupRecommender(elsa, aggregator)
    
    
    recommend(args, group_recommender, elsa, groups, interactions)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)