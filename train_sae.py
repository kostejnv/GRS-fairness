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

TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.info(f'Training SAE model')
device = Utils.set_device()
logging.info(f'Device: {device}')

def parse_arguments():
    parser = argparse.ArgumentParser() # TODO: Add description
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # dataset
    parser.add_argument('--dataset', type=str, default='LastFM1k', help='Dataset to use. For now, only "LastFM1k" and "EchoNest" are supported')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    # training details
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--early_stop', type=int, default=50, help='Number of epochs to wait for improvement before stopping')
    # optimizer
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    # model
    parser.add_argument('--model', type=str, default='TopKSAE', help='Model to use (BasicSAE, TopKSAE)')
    parser.add_argument('--embedding_dim', type=int, default=2048, help='Number of factors for the model')
    parser.add_argument('--top_k', type=int, default=64, help='Top k parameter for TopKSAE')
    parser.add_argument("--reconstruction_loss", type=str, default="Cosine", help="Reconstruction loss (L2 or Cosine)")
    parser.add_argument("--l1_coef", type=float, default=3e-4, help="L1 loss coefficient (BasicSAE, TopKSAE)")
    # base model
    parser.add_argument("--base_run_id", type=str, default='32b65a3a9edf4ff4b46e9d8385d93bc4', help="Run ID of the base model")
    
    # evaluate
    parser.add_argument('--target_ratio', type=float, default=0.2, help='Ratio of target interactions')
    return parser.parse_args()

def train(args, model:SAE, base_model:ELSA, optimizer, train_csr, valid_csr, test_csr):
    dataset = args.dataset
    nr_epochs = args.epochs
    batch_size = args.batch_size
    early_stop = args.early_stop
    
    mlflow.set_experiment(f'Sparse_{dataset}')
    mlflow.set_experiment_tags({
        'dataset': args.dataset,
        'task': 'dense_training',
        'mlflow.note.content': f'This experiments trains an dense user embedding for the {dataset}.'
    })
    
    if args.model == 'TopKSAE':
        run_name = f'{args.model}_{args.embedding_dim}_{args.top_k}_{TIMESTAMP}'
    else:
        run_name = f'{args.model}_{args.embedding_dim}_{TIMESTAMP}'

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(vars(args))
        
        train_interaction_dataloader = DataLoader(train_csr, batch_size, device, shuffle=False)
        valid_interaction_dataloader = DataLoader(valid_csr, batch_size, device, shuffle=False)
        
        train_user_embeddings = np.vstack(
            [
                base_model.encode(batch).detach().cpu().numpy()
                for batch in tqdm(train_interaction_dataloader, desc="Computing user embeddings from train interactions")
            ]
        )
        
        val_user_embeddings = np.vstack(
            [
                base_model.encode(batch).detach().cpu().numpy()
                for batch in tqdm(valid_interaction_dataloader, desc="Computing user embeddings from train interactions")
            ]
        )
        
        train_embeddings_dataloader = DataLoader(train_user_embeddings, batch_size, device, shuffle=True)
        valid_embeddings_dataloader = DataLoader(val_user_embeddings, batch_size, device, shuffle=False)
        
        if early_stop > 0:
            best_epoch = 0
            epochs_without_improvement = 0
            best_sim = 0
            best_optimizer = deepcopy(optimizer)
            best_model = deepcopy(model)
        
        for epoch in range(1, nr_epochs+1):
            train_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": []}
            model.train()
            
            pbar = tqdm(train_embeddings_dataloader, desc=f'Epoch {epoch}/{nr_epochs}')
            for batch in pbar: # train one batch
                losses = model.train_step(optimizer, batch)
                pbar.set_postfix({'train_loss': losses['Loss'].cpu().item()})
                
                for key, val in train_losses.items():
                    val.append(losses[key].item())                    
                
            for key, val in train_losses.items():
                mlflow.log_metric(f'loss/{key}/train', float(np.mean(val)), step=epoch)
                
            # Evaluate
            model.eval()
            # loss
            valid_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": []}
            for batch in valid_embeddings_dataloader:
                losses = model.compute_loss_dict(batch)
                for key, val in losses.items():
                    valid_losses[key].append(val.item())
            for key, val in valid_losses.items():
                mlflow.log_metric(f'loss/{key}/valid', float(np.mean(val)), step=epoch)
                
            # metrics
            valid_metrics = Utils.evaluate_sparse_encoder(base_model, model, valid_csr, args.target_ratio, batch_size, device)
            for key, val in valid_metrics.items():
                mlflow.log_metric(f'{key}/valid', val, step=epoch)
            
            logging.info(f'Valid metrics - Loss: {float(np.mean(valid_losses["Loss"])):.4f} - Cosine: {valid_metrics["CosineSim"]:.4f} - NDCG20 Degradation: {valid_metrics["NDCG20_Degradation"]:.4f}')
            
            if early_stop > 0:
                if valid_metrics['CosineSim'] > best_sim:
                    best_sim = valid_metrics['CosineSim']
                    best_optimizer = deepcopy(optimizer)
                    best_model = deepcopy(model)
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stop:
                        logging.info(f'Early stopping at epoch {epoch}')
                        break
        if early_stop > 0:
            logging.info(f'Loading best model from epoch {best_epoch}')
            model = best_model
            optimizer = best_optimizer
            
        test_metrics = Utils.evaluate_sparse_encoder(base_model, model, test_csr, args.target_ratio, batch_size, device)
        for key, val in test_metrics.items():
            mlflow.log_metric(f'{key}/test', val, step=epoch)
        logging.info(f'Test metrics - Cosine: {test_metrics["CosineSim"]:.4f} - NDCG20 Degradation: {test_metrics["NDCG20_Degradation"]:.4f}')
        
        # Save model
        temp_path = 'checkpoint.ckpt'
        Utils.save_checkpoint(model, optimizer, temp_path)
        mlflow.log_artifact(temp_path)
        mlflow.log_artifact('models/sae.py')
        os.remove(temp_path)
        logging.info('Model successfully saved')
                
def main(args):
    Utils.set_seed(args.seed)
    
    # log info about base model
    base_model_run = mlflow.get_run(args.base_run_id)
    
    base_params = base_model_run.data.params
    artifact_path = base_model_run.info.artifact_uri.replace('file://', '') # type: ignore
    
    assert base_params['dataset'] == args.dataset, 'Base model dataset does not match current dataset'
    
    args.base_model = base_params['model']
    args.base_factors = int(base_params['factors'])
    args.base_min_user_interactions = int(base_params['min_user_interactions'])
    args.base_min_item_interactions = int(base_params['min_item_interactions'])
    args.base_users = int(base_params['users'])
    args.base_items = int(base_params['items'])
    args.expansion_ratio = args.embedding_dim / args.base_factors
    
    #Load dataset
    logging.info(f'Loading {args.dataset}')
    match args.dataset:
        case 'EchoNest':
            dataset_loader = EchoNestLoader()
        case 'LastFM1k':
            dataset_loader = LastFm1kLoader()
        case _:
            raise ValueError(f'Dataset {args.dataset} not supported. Check typos.')
    dataset_loader.prepare(args)
    
    args.min_user_interactions = dataset_loader.MIN_USER_INTERACTIONS
    args.min_item_interactions = dataset_loader.MIN_ITEM_INTERACTIONS
    args.users = len(dataset_loader.users)
    args.items = len(dataset_loader.items)
    
    assert args.items == args.base_items, 'Number of items in dataset does not match base model'
    assert args.users == args.base_users, 'Number of users in dataset does not match base model'
    
    train_csr = dataset_loader.train_csr
    valid_csr = dataset_loader.valid_csr
    test_csr = dataset_loader.test_csr
    
    base_model = ELSA(args.base_items, args.base_factors).to(device)
    base_optimizer = torch.optim.Adam(base_model.parameters())
    Utils.load_checkpoint(base_model, base_optimizer, f'{artifact_path}/checkpoint.ckpt')
    base_model.eval()
    
    match args.model:
        case 'BasicSAE':
            model = BasicSAE(args.base_factors, args.embedding_dim, args.reconstruction_loss, l1_coef=args.l1_coef).to(device)
        case 'TopKSAE':
            model = TopKSAE(args.base_factors, args.embedding_dim, args.reconstruction_loss, l1_coef=args.l1_coef, k=args.top_k).to(device)
        case _:
            raise ValueError(f'Model {args.model} not supported. Check typos.')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    train(args, model, base_model, optimizer, train_csr, valid_csr, test_csr)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)