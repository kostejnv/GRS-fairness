import argparse
import logging
import sys
import torch
from datasets import EchoNestLoader, LastFm1kLoader, DataLoader, MovieLensLoader
from models import ELSA, ELSAWithSAE, BasicSAE, TopKSAE, SAE
import mlflow
import numpy as np
import random
import os
import datetime
from tqdm import tqdm
from utils import Utils
from copy import deepcopy

TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.info(f'Analyzing Group Embedding Intersection - {TIMESTAMP}')
device = Utils.set_device()
logging.info(f'Device: {device}')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to use. For now, only "LastFM1k" and "EchoNest" and "MovieLens" are supported')
    parser.add_argument("--sae_run_id", type=str, help="Run ID of the analyzed SAE model")
    parser.add_argument("--group_type", type=str, help="Type of group to analyze. Options: 'sim', 'div', '21', 'random'")
    parser.add_argument("--group_size", type=int, default=3, help="Size of the group to analyze")
    parser.add_argument("--user_set", type=str, default='train', help="User set from which the groups where sampled (full, test, train)")
    # stable parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    
    return parser.parse_args()


def main(args):
    assert args.group_size == 3, 'Only group size 3 is supported for now'
    assert args.group_type in ['sim', 'div', '21', 'random'], 'Group type not supported'
    
    # load sae model
    sae_run = mlflow.get_run(args.sae_run_id)
    sae_params = sae_run.data.params
    
    # log parameters
    args.embedding_dim = int(sae_params['embedding_dim'])
    args.top_k = sae_params.get('top_k', None)
    args.sample_users = sae_params['sample_users']
    args.sae_model = sae_params['model']
    args.sae_run_id = args.sae_run_id
    
    sae_artifact_path = sae_run.info.artifact_uri
    sae_artifact_path = './' + sae_artifact_path[sae_artifact_path.find('mlruns'):]
    
    # load base model
    base_model_run = mlflow.get_run(sae_params['base_run_id'])
    base_model_params = base_model_run.data.params
    
    
    assert sae_params['dataset'] == args.dataset, 'Base model dataset does not match current dataset'
    
    base_artifact_path = base_model_run.info.artifact_uri
    base_artifact_path = './' + base_artifact_path[base_artifact_path.find('mlruns'):]
    
    # Load groups
    if args.group_type == 'sim':
        filename = f'similar_{args.group_size}.npy'
    elif args.group_type == 'div':
        filename = f'divergent_{args.group_size}.npy'
    elif args.group_type == '21':
        filename = f'opposing_2_1.npy'
    elif args.group_type == 'random':
        filename = f'random_{args.group_size}.npy'
            
    if args.dataset == 'LastFM1k':
        dataset_loader = LastFm1kLoader()
    elif args.dataset == 'EchoNest':
        dataset_loader = EchoNestLoader()
    elif args.dataset == 'MovieLens':
        dataset_loader = MovieLensLoader()
    dataset_loader.prepare(args)

    path = f'data/synthetic_groups/{args.dataset}/'
    if args.user_set == 'full':
        path += 'full'
    elif args.user_set == 'train':
        path += 'train'
    elif args.user_set == 'test':
        path += 'test'
    else:
        raise ValueError(f'User set {args.user_set} not supported. Check typo.')
            
    path += f'/{filename}'
    if not os.path.exists(path):
        raise FileNotFoundError(f'File {path} not found')
    
    groups = np.load(path, allow_pickle=True)
    groups = np.vectorize(lambda x: np.argwhere(dataset_loader.users == x))(groups)
    logging.info(f'Groups with shape {groups.shape} loaded from {path}')
    
    # Load models
    base_factors = int(base_model_params['factors'])
    base_items = int(base_model_params['items'])
    
    sae_embedding_dim = int(sae_params['embedding_dim'])
    sae_reconstruction_loss = sae_params['reconstruction_loss']
    sae_l1_coef = float(sae_params['l1_coef'])
    sae_top_k = int(sae_params['top_k'])
    base_factors = int(sae_params['base_factors'])
    sae_topk_aux = int(sae_params.get('topk_aux', 0))
    sae_n_batches_to_dead = int(sae_params.get('n_batches_to_dead', 0))
    sae_normalize = True if sae_params.get('normalize', 'False') == 'True' else False
    sae_auxiliary_coef = float(sae_params.get('auxiliary_coef', 0))
    sae_contrastive_coef = float(sae_params.get('contrastive_coef', 0))
    sae_reconstruction_coef = float(sae_params.get('reconstruction_coef', 1))
    
    cfg = {
        'reconstruction_loss': sae_reconstruction_loss,
        "input_dim": base_factors,
        "embedding_dim": sae_embedding_dim,
        "l1_coef": sae_l1_coef,
        "k": sae_top_k,
        "device": device,
        "topk_aux": sae_topk_aux,
        "n_batches_to_dead": sae_n_batches_to_dead,
        "normalize": sae_normalize,
        "auxiliary_coef": sae_auxiliary_coef,
        "contrastive_coef": sae_contrastive_coef,
        "reconstruction_coef": sae_reconstruction_coef,
    }
    
    elsa = ELSA(base_items, base_factors)
    optimizer = torch.optim.Adam(elsa.parameters())
    Utils.load_checkpoint(elsa, optimizer, f'{base_artifact_path}/checkpoint.ckpt', device)
    elsa.to(device)
    elsa.eval()
    logging.info(f'ELSA model loaded from {base_artifact_path}')
    
    if sae_params['model'] == 'BasicSAE':
        sae = BasicSAE(base_factors, sae_embedding_dim, cfg).to(device)
    elif sae_params['model'] == 'TopKSAE':
        sae = TopKSAE(base_factors, sae_embedding_dim, cfg).to(device)
    else:
        raise ValueError(f'Model {sae_params["model"]} not supported. Check typos.')
    
    sae_optimizer = torch.optim.Adam(sae.parameters())
    Utils.load_checkpoint(sae, sae_optimizer, f'{sae_artifact_path}/checkpoint.ckpt', device)
    sae.to(device)
    sae.eval()
    
    logging.info(f'SAE model loaded from {sae_artifact_path}')
    
    # Start analysis
    logging.info('Starting group analysis')
    
    mlflow.set_experiment(f'GEI_{args.dataset}')
    mlflow.set_experiment_tags({
        'dataset': args.dataset,
        'task': 'intersection_analysis',
        'mlflow.note.content': f'This experiments analyzes the intersection of group embeddings in the {args.dataset} dataset.'
    })
    run_name = f'{args.sae_model[:3]}_{args.group_type}_{args.group_size}_{args.embedding_dim}_{args.top_k}_{TIMESTAMP}'
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(vars(args))
        
        common_features = [] # nr. of features that is shared between the whole group
        at_least_2_features = [] # nr. of features that is shared between at least 2 users in the group
        at_least_1_feature = [] # nr. of features that has at least 1 user from group
        
        
        for group_idxs in groups:
            group_interactions = torch.tensor(dataset_loader.csr_interactions[group_idxs].toarray(), device=device)
            group_embeddings = sae.encode(elsa.encode(group_interactions))[0].to('cpu')
            
            binarized = group_embeddings > 0
            common_features.append(binarized.all(dim=0).sum())
            at_least_2_features.append((binarized.sum(dim=0) >= 2).sum())
            at_least_1_feature.append((binarized.sum(dim=0) >= 1).sum())
            
        mlflow.log_metrics({
            'common_features/mean': float(np.mean(common_features)),
            'common_features/std': float(np.std(common_features)),
            'common_features/min': float(np.min(common_features)),
            'common_features/max': float(np.max(common_features)),
            
            'at_least_2_features/mean': float(np.mean(at_least_2_features)),
            'at_least_2_features/std': float(np.std(at_least_2_features)),
            'at_least_2_features/min': float(np.min(at_least_2_features)),
            'at_least_2_features/max': float(np.max(at_least_2_features)),
            
            'at_least_1_feature/mean': float(np.mean(at_least_1_feature)),
            'at_least_1_feature/std': float(np.std(at_least_1_feature)),
            'at_least_1_feature/min': float(np.min(at_least_1_feature)),
            'at_least_1_feature/max': float(np.max(at_least_1_feature)),
        })

if __name__ == '__main__':
    args = parse_arguments()
    main(args)