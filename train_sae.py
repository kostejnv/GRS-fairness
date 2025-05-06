import argparse
import logging
import sys
import torch
from datasets import EchoNestLoader, LastFm1kLoader, DataLoader, MovieLensLoader
from models import ELSA, ELSAWithSAE, BasicSAE, TopKSAE, SAE, BatchTopKSAE
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LastFM1k', help='Dataset to use. For now, only "LastFM1k" and "EchoNest" and "MovieLens" are supported')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--early_stop', type=int, default=10, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=2048, help='Number of factors for the model')
    parser.add_argument('--top_k', type=int, default=128, help='Top k parameter for TopKSAE')
    parser.add_argument("--base_run_id", type=str, default='4a43996d7eec489183ad0d6b0c00d935', help="Run ID of the base model")
    parser.add_argument("--sample_users", action='store_true', default=False, help="Choose randomly 0.5 - 1.0 of the users interactions")
    # stable parameters
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    parser.add_argument('--model', type=str, default='BatchTopKSAE', help='Model to use (BasicSAE, TopKSAE, BatchTopKSAE)')
    parser.add_argument("--reconstruction_loss", type=str, default="Cosine", help="Reconstruction loss (L2 or Cosine)")
    parser.add_argument("--auxiliary_coef", type=float, default=1/32, help="Auxiliary loss coefficient (BasicSAE, TopKSAE)")
    parser.add_argument("--contrastive_coef", type=float, default=0.3, help="Contrastive loss coefficient (BasicSAE, TopKSAE)")
    parser.add_argument("--n_batches_to_dead", type=int, default=5, help="Number of batches to wait before optimizing the dead neurons (BasicSAE, TopKSAE)")
    parser.add_argument("--normalize", action='store_true', help="Normalize the sparse embedding (BasicSAE, TopKSAE)")
    parser.add_argument("--topk_aux", type=int, default=512, help="Top k for auxiliary loss (BasicSAE, TopKSAE)")
    parser.add_argument("--l1_coef", type=float, default=3e-4, help="L1 loss coefficient (BasicSAE, TopKSAE)")
    parser.add_argument('--target_ratio', type=float, default=0.2, help='Ratio of target interactions')
    parser.add_argument('--evaluate_every', type=int, default=10, help='Evaluate every n epochs')
    parser.add_argument('--note', type=str, default='', help='Note for the experiment')
    
    return parser.parse_args()

def train(args, model:SAE, base_model:ELSA, optimizer, train_csr, valid_csr, test_csr):
    dataset = args.dataset
    nr_epochs = args.epochs
    batch_size = args.batch_size
    early_stop = args.early_stop
    evaluate_every = args.evaluate_every
    
    mlflow.set_experiment(f'Sparse_{dataset}')
    mlflow.set_experiment_tags({
        'dataset': args.dataset,
        'task': 'dense_training',
        'mlflow.note.content': f'This experiments trains an dense user embedding for the {dataset}.'
    })
    
    if args.model in ['TopKSAE', 'BatchTopKSAE']:
        run_name = f'{args.model}_{args.embedding_dim}_{args.top_k}_{TIMESTAMP}'
    else:
        run_name = f'{args.model}_{args.embedding_dim}_{TIMESTAMP}'

    def sampled_interactions(batch, ratio = 0.8):
        mask = torch.rand_like(batch) < ratio
        return batch.clone() * mask
    
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
        train_embeddings_dataloader = DataLoader(train_user_embeddings, batch_size, device, shuffle=True)
        
        val_user_embeddings = np.vstack(
            [
                base_model.encode(sampled_interactions(batch)).detach().cpu().numpy()
                for batch in tqdm(valid_interaction_dataloader, desc="Computing user embeddings from train interactions")
            ]
        )
        valid_embeddings_dataloader = DataLoader(val_user_embeddings, batch_size, device, shuffle=False)
        
        
        val_positive_user_embeddings = np.vstack(
            [
                base_model.encode(sampled_interactions(batch)).detach().cpu().numpy()
                for batch in tqdm(valid_interaction_dataloader, desc="Computing augumentation of user embeddings from train interactions")
            ]
        )
        val_positive_embeddings_dataloader = DataLoader(val_positive_user_embeddings, batch_size, device, shuffle=False)
        
        val_user_embeddings = torch.tensor(val_user_embeddings, device=device)
        
        def train_epoch_from_interactions():
            train_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": []}
            model.train()
            
            pbar = tqdm(train_interaction_dataloader, desc=f'Epoch {epoch}/{nr_epochs}')
            for batched_interactions in pbar: # train one batch
                if args.contrastive_coef > 0:
                    positive_batch = sampled_interactions(batched_interactions, ratio=0.5)
                    positive_batch = base_model.encode(positive_batch).detach()
                else:
                    positive_batch = None
                if args.sample_users:
                    batched_interactions = sampled_interactions(batched_interactions)
                    
                embedding_batch = base_model.encode(batched_interactions).detach()
                
                losses = model.train_step(optimizer, embedding_batch, positive_batch)
                pbar.set_postfix({'train_loss': losses['Loss'].cpu().item()})
                for key, val in train_losses.items():
                    val.append(losses[key].item())
            return train_losses
                    
        def train_epoch_from_embeddings(): # faster but cannot use contrastive loss or sample users
            train_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": []}
            model.train()
            pbar = tqdm(train_embeddings_dataloader, desc=f'Epoch {epoch}/{nr_epochs}')
            for batched_embeddings in pbar:
                losses = model.train_step(optimizer, batched_embeddings, None)
                pbar.set_postfix({'train_loss': losses['Loss'].cpu().item()})
                for key, val in train_losses.items():
                    val.append(losses[key].item())
                    
            return train_losses
                    
        if early_stop > 0:
            best_epoch = 0
            epochs_without_improvement = 0
            best_sim = np.inf
            best_optimizer = deepcopy(optimizer)
            best_model = deepcopy(model)
            
        are_interactions_needed = args.sample_users or args.contrastive_coef > 0
        for epoch in range(1, nr_epochs+1):
            if are_interactions_needed:
                train_losses = train_epoch_from_interactions()
            else:
                train_losses = train_epoch_from_embeddings()
            
            for key, val in train_losses.items():
                mlflow.log_metric(f'loss/{key}/train', float(np.mean(val)), step=epoch)
                    
            if epoch % evaluate_every == 0:
                # Evaluate
                model.eval()
                # loss
                valid_losses = {"Loss": [], "L2": [], "L1": [], "L0": [], "Cosine": [], "Auxiliary": [], "Contrastive": []}
                for embedding, positive_embedding in zip(valid_embeddings_dataloader, val_positive_embeddings_dataloader):
                    losses = model.compute_loss_dict(embedding, positive_embedding)
                    for key, val in losses.items():
                        valid_losses[key].append(val.item())
                for key, val in valid_losses.items():
                    mlflow.log_metric(f'loss/{key}/valid', float(np.mean(val)), step=epoch)
                    
                # metrics
                valid_metrics = Utils.evaluate_sparse_encoder(base_model, model, valid_csr, args.target_ratio, batch_size, device, seed=args.seed)
                for key, val in valid_metrics.items():
                    mlflow.log_metric(f'{key}/valid', val, step=epoch)
                
                logging.info(f'Valid metrics - Loss: {float(np.mean(valid_losses["Loss"])):.4f} - Cosine: {valid_metrics["CosineSim"]:.4f} - NDCG20 Degradation: {valid_metrics["NDCG20_Degradation"]:.4f}, Contrastive: {float(np.mean(valid_losses["Contrastive"])):.4f}')
                
                valid_loss = float(np.mean(valid_losses["Loss"]))
                if early_stop > 0:
                    if valid_loss < best_sim:
                        best_sim = valid_loss
                        best_optimizer = deepcopy(optimizer)
                        best_model = deepcopy(model)
                        best_epoch = epoch
                        epochs_without_improvement = 0
                    else:
                        if epochs_without_improvement >= early_stop:
                            logging.info(f'Early stopping at epoch {epoch}')
                            break
            epochs_without_improvement += 1
        if early_stop > 0:
            logging.info(f'Loading best model from epoch {best_epoch}')
            model = best_model
            optimizer = best_optimizer
            
        test_metrics = Utils.evaluate_sparse_encoder(base_model, model, test_csr, args.target_ratio, batch_size, device, seed=args.seed)
        for key, val in test_metrics.items():
            mlflow.log_metric(f'{key}/test', val, step=epoch)
        logging.info(f'Test metrics - Cosine: {test_metrics["CosineSim"]:.4f} - NDCG20 Degradation: {test_metrics["NDCG20_Degradation"]:.4f}')
        
        # Save model
        temp_path = './checkpoint.ckpt'
        Utils.save_checkpoint(model, optimizer, temp_path)
        print(f'Saving model to {temp_path}')
        mlflow.log_artifact(temp_path)
        mlflow.log_artifact('models/sae.py')
        # os.remove(temp_path)
        logging.info('Model successfully saved')
        
        cfg = {
            'reconstruction_loss': args.reconstruction_loss,
            "topk_aux": args.topk_aux,
            "n_batches_to_dead": args.n_batches_to_dead,
            "l1_coef": args.l1_coef,
            "k": args.top_k,
            "device": device,
            "normalize": args.normalize,
            "auxiliary_coef": args.auxiliary_coef,
            "contrastive_coef": args.contrastive_coef,
            "reconstruction_coef": args.reconstruction_coef,
        }
        
        if args.model == 'BasicSAE':
            model = BasicSAE(args.base_factors, args.embedding_dim, cfg).to(device)
        elif args.model == 'TopKSAE':
            model = TopKSAE(args.base_factors, args.embedding_dim, cfg).to(device)
        elif args.model == 'BatchTopKSAE':
            model = BatchTopKSAE(args.base_factors, args.embedding_dim, cfg).to(device)
        else:
            raise ValueError(f'Model {args.model} not supported. Check typos.')
        optimizer = torch.optim.Adam(model.parameters())
        Utils.load_checkpoint(model, optimizer, temp_path, device)
        print(model.threshold)
        
        test_metrics = Utils.evaluate_sparse_encoder(base_model, model, test_csr, args.target_ratio, batch_size, device, seed=args.seed)
        for key, val in test_metrics.items():
            mlflow.log_metric(f'{key}/test', val, step=epoch)
        logging.info(f'Test metrics - Cosine: {test_metrics["CosineSim"]:.4f} - NDCG20 Degradation: {test_metrics["NDCG20_Degradation"]:.4f}')
        print(test_metrics)
        print(model.encoder_b)
                
def main(args):
    Utils.set_seed(args.seed)
    
    # log info about base model
    base_model_run = mlflow.get_run(args.base_run_id)
    
    base_params = base_model_run.data.params
    artifact_path = base_model_run.info.artifact_uri
    # remove all before mlruns
    artifact_path = './' + artifact_path[artifact_path.find('mlruns'):]
    
    assert base_params['dataset'] == args.dataset, 'Base model dataset does not match current dataset'
    
    logging.info(f'Params: {vars(args)}')
    
    args.base_model = base_params['model']
    args.base_factors = int(base_params['factors'])
    args.base_min_user_interactions = int(base_params['min_user_interactions'])
    args.base_min_item_interactions = int(base_params['min_item_interactions'])
    args.base_users = int(base_params['users'])
    args.base_items = int(base_params['items'])
    args.expansion_ratio = args.embedding_dim / args.base_factors
    args.reconstruction_coef = 1 - (args.auxiliary_coef + args.contrastive_coef + args.l1_coef)
    
    # Load dataset
    logging.info(f'Loading {args.dataset}')
    if args.dataset == 'EchoNest':
        dataset_loader = EchoNestLoader()
    elif args.dataset == 'LastFM1k':
        dataset_loader = LastFm1kLoader()
    elif args.dataset == 'MovieLens':
        dataset_loader = MovieLensLoader()
    else:
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
    print(test_csr.sum())
    
    base_model = ELSA(args.base_items, args.base_factors)
    base_optimizer = torch.optim.Adam(base_model.parameters())
    Utils.load_checkpoint(base_model, base_optimizer, f'{artifact_path}/checkpoint.ckpt', device)
    base_model.to(device)
    base_model.eval()
    
    cfg = {
        'reconstruction_loss': args.reconstruction_loss,
        "topk_aux": args.topk_aux,
        "n_batches_to_dead": args.n_batches_to_dead,
        "l1_coef": args.l1_coef,
        "k": args.top_k,
        "device": device,
        "normalize": args.normalize,
        "auxiliary_coef": args.auxiliary_coef,
        "contrastive_coef": args.contrastive_coef,
        "reconstruction_coef": args.reconstruction_coef,
    }
    if args.model == 'BasicSAE':
        model = BasicSAE(args.base_factors, args.embedding_dim, cfg).to(device)
    elif args.model == 'TopKSAE':
        model = TopKSAE(args.base_factors, args.embedding_dim, cfg).to(device)
    elif args.model == 'BatchTopKSAE':
        model = BatchTopKSAE(args.base_factors, args.embedding_dim, cfg).to(device)
    else:
        raise ValueError(f'Model {args.model} not supported. Check typos.')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    train(args, model, base_model, optimizer, train_csr, valid_csr, test_csr)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)