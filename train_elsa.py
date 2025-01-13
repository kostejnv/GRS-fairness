import argparse
import logging
import sys
import torch
from datasets import EchoNestLoader, LastFm1kLoader, DataLoader
from models import ELSA
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

device = Utils.set_device()
logging.info(f'Device: {device}')

def parse_arguments():
    parser = argparse.ArgumentParser() # TODO: Add description
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # dataset
    parser.add_argument('--dataset', type=str, default='EchoNest', help='Dataset to use. For now, only "LastFM1k" and "EchoNest" are supported')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    # training details
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    # model
    parser.add_argument('--factors', type=int, default=1024, help='Number of factors for the model')
    # evaluate
    parser.add_argument('--target_ratio', type=float, default=0.2, help='Ratio of target interactions')
    parser.add_argument('--early_stop', type=int, default=5, help='Number of epochs to wait for improvement before stopping')
    return parser.parse_args()

def train(args, model: ELSA, optimizer, train_csr, valid_csr, test_csr):
    dataset = args.dataset
    nr_epochs = args.epochs
    batch_size = args.batch_size
    early_stop = args.early_stop
    
    mlflow.set_experiment(f'ELSA_{dataset}')
    mlflow.set_experiment_tags({
        'dataset': args.dataset,
        'recommendation_type': 'user',
        'model': 'ELSA',
        'mlflow.note.content': f'This experiments trains an ELSA model on the {dataset} dataset'
    })
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        
        train_dataloader = DataLoader(train_csr, batch_size, device)
        valid_dataloader = DataLoader(valid_csr, batch_size, device, shuffle=False)
        
        if early_stop > 0:
            best_epoch = 0
            epochs_without_improvement = 0
            best_ndcg = 0
            best_optimizer = deepcopy(optimizer)
            best_model = deepcopy(model)
        
        for epoch in range(1, nr_epochs+1):
            train_losses = []
            model.train()
            
            pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{nr_epochs}')
            for batch in pbar: # train one batch
                losses = model.train_step(optimizer, batch)
                train_losses.append(losses['Loss'].item())
                pbar.set_postfix({'train_loss': losses['Loss'].cpu().item()})
            mlflow.log_metric('loss/train', float(np.mean(train_losses)), step=epoch)
            # Evaluate
            model.eval()
            valid_metrics = Utils.evaluate(model, valid_csr, args.target_ratio, batch_size, device)
            valid_metrics['loss'] = float(np.mean([model.compute_loss_dict(batch)['Loss'].item() for batch in valid_dataloader]))
            for key, val in valid_metrics.items():
                mlflow.log_metric(f'{key}/valid', val, step=epoch)
            logging.info(f'Epoch {epoch}/{nr_epochs} - Loss: {valid_metrics["loss"]:.4f} - R@20: {valid_metrics["R20"]:.4f} - NDCG20: {valid_metrics["NDCG20"]:.4f}')
            
            if early_stop > 0:
                if valid_metrics['NDCG20'] > best_ndcg:
                    best_ndcg = valid_metrics['NDCG20']
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
            
        test_metrics = Utils.evaluate(model, test_csr, args.target_ratio, batch_size, device)
        for key, val in test_metrics.items():
            mlflow.log_metric(f'{key}/test', val)
        logging.info(f'Test metrics - R@20: {test_metrics["R20"]:.4f} - NDCG20: {test_metrics["NDCG20"]:.4f}')
        
        # Save model
        temp_path = 'checkpoint.ckpt'
        Utils.save_checkpoint(model, optimizer, temp_path)
        mlflow.log_artifact(temp_path)
        mlflow.log_artifact('models/elsa.py')
        os.remove(temp_path)
        logging.info('Model successfully saved')
                
def main(args):
    
    Utils.set_seed(args.seed)
    
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
    
    train_csr = dataset_loader.train_csr
    valid_csr = dataset_loader.valid_csr
    test_csr = dataset_loader.test_csr
    
    item_cnt = len(dataset_loader.items)
    model = ELSA(input_dim=item_cnt, embedding_dim=args.factors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    train(args, model, optimizer, train_csr, valid_csr, test_csr)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)