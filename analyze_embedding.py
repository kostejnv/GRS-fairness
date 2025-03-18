import argparse
import logging
import sys
import torch
from datasets import EchoNestLoader, LastFm1kLoader, DataLoader
from models import ELSA, ELSAWithSAE, BasicSAE, TopKSAE, SAE
import mlflow
import numpy as np
import random
import tqdm
import os
import datetime
from utils import Utils
from copy import deepcopy
from plotly import graph_objects as go

TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.info(f'Analyzing Embedding - {TIMESTAMP}')
device = Utils.set_device()
# device = torch.device('cpu')
logging.info(f'Device: {device}')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to use. For now, only "LastFM1k" and "EchoNest" are supported')
    parser.add_argument("--sae_run_id", type=str, help="Run ID of the analyzed SAE model")
    parser.add_argument("--user_set", type=str, default='full', help="User set to analyze (full, test, train)")
    parser.add_argument("--user_sample", type=int, default=10_000, help="Number of users to sample")
    # stable parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    
    return parser.parse_args()


def main(args):
    
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
            
    if args.dataset == 'LastFM1k':
        dataset_loader = LastFm1kLoader()
    elif args.dataset == 'EchoNest':
        dataset_loader = EchoNestLoader()
            
    dataset_loader.prepare(args)
    
    # load interactions
    if args.user_set == 'full':
        csr_interactions = dataset_loader.csr_interactions
    elif args.user_set == 'train':
        csr_interactions = dataset_loader.train_csr
    elif args.user_set == 'test':
        csr_interactions = dataset_loader.test_csr
    else:
        raise ValueError(f'User set {args.user_set} not supported. Check typo.')

    interactions_batches = DataLoader(csr_interactions, batch_size=1024, device=device, shuffle=False)

    # Load models
    base_factors = int(base_model_params['factors'])
    base_items = int(base_model_params['items'])
    
    sae_embedding_dim = int(sae_params['embedding_dim'])
    sae_reconstruction_loss = sae_params['reconstruction_loss']
    sae_l1_coef = float(sae_params['l1_coef'])
    sae_top_k = int(sae_params['top_k'])
    
    elsa = ELSA(base_items, base_factors)
    optimizer = torch.optim.Adam(elsa.parameters())
    Utils.load_checkpoint(elsa, optimizer, f'{base_artifact_path}/checkpoint.ckpt', device)
    elsa.to(device)
    elsa.eval()
    logging.info(f'ELSA model loaded from {base_artifact_path}')
    
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
    
    # Start analysis
    logging.info('Starting embedding analysis')
    
    mlflow.set_experiment(f'EA_{args.dataset}_{args.user_set}')
    mlflow.set_experiment_tags({
        'dataset': args.dataset,
        'task': 'embedding_analysis',
        'user_set': args.user_set,
        'mlflow.note.content': f'This experiments analyzes the sparse user embeddings for the {args.user_set} user set in the {args.dataset} dataset.'
    })
    run_name = f'{args.sae_model}_{args.embedding_dim}_{args.top_k}'
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(vars(args))
        
        # create user embeddings
        batches_embeddings = []
        for batch in tqdm.tqdm(interactions_batches, desc='Creating user embeddings'):
            batch_embeddings = sae.encode(elsa.encode(batch))[0]
            batches_embeddings.append(batch_embeddings.detach().to('cpu'))
        user_embeddings = torch.cat(batches_embeddings)
        
        # filter all zero values
        user_embeddings_tmp = user_embeddings.numpy().tolist()
        user_embeddings_tmp = [[x for x in row if x != 0] for row in user_embeddings_tmp]
        for row in user_embeddings_tmp:
            row.extend([0.0] * (int(args.top_k) - len(row)))
        user_embeddings = torch.tensor(user_embeddings_tmp, device=device)
        
        max_embedding = torch.max(user_embeddings, dim=1).values
        min_embedding = torch.min(user_embeddings, dim=1).values
        mean_embedding = torch.mean(user_embeddings, dim=1)
        std_embedding = torch.std(user_embeddings, dim=1)
        median_embedding = torch.median(user_embeddings, dim=1).values
        
        mlflow.log_metrics({
            'max_embedding/max': float(torch.max(max_embedding)),
            'max_embedding/min': float(torch.min(max_embedding)),
            'max_embedding/mean': float(torch.mean(max_embedding)),
            'max_embedding/std': float(torch.std(max_embedding)),
            'max_embedding/median': float(torch.median(median_embedding)),
            
            'min_embedding/max': float(torch.max(min_embedding)),
            'min_embedding/min': float(torch.min(min_embedding)),
            'min_embedding/mean': float(torch.mean(min_embedding)),
            'min_embedding/std': float(torch.std(min_embedding)),
            'min_embedding/median': float(torch.median(median_embedding)),
            
            'mean_embedding/max': float(torch.max(mean_embedding)),
            'mean_embedding/min': float(torch.min(mean_embedding)),
            'mean_embedding/mean': float(torch.mean(mean_embedding)),
            'mean_embedding/std': float(torch.std(mean_embedding)),
            'mean_embedding/median': float(torch.median(median_embedding)),
            
            'std_embedding/max': float(torch.max(std_embedding)),
            'std_embedding/min': float(torch.min(std_embedding)),
            'std_embedding/mean': float(torch.mean(std_embedding)),
            'std_embedding/std': float(torch.std(std_embedding)),
            'std_embedding/median': float(torch.median(median_embedding)),
            
            'median_embedding/max': float(torch.max(median_embedding)),
            'median_embedding/min': float(torch.min(median_embedding)),
            'median_embedding/mean': float(torch.mean(median_embedding)),
            'median_embedding/std': float(torch.std(median_embedding)),
            'median_embedding/median': float(torch.median(median_embedding)),
        })
        
        
        # plot histograms
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=user_embeddings.flatten().to('cpu').detach().numpy()))
        fig.update_layout(title_text='User Embedding All Value Histogram', xaxis_title='Value', yaxis_title='Frequency')
        mlflow.log_figure(fig, 'all_values_histogram.html')
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=max_embedding.to('cpu').detach().numpy()))
        fig.update_layout(title_text='User Embedding Max Value Histogram', xaxis_title='Value', yaxis_title='Frequency')
        mlflow.log_figure(fig, 'max_values_histogram.html')
        

if __name__ == '__main__':
    args = parse_arguments()
    main(args)