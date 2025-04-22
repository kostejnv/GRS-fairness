import argparse
import logging
import sys
import torch
from datasets import EchoNestLoader, LastFm1kLoader, DataLoader
from models import ELSA, ELSAWithSAE, BasicSAE, TopKSAE, SAE
from popularity import Popularity
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
from group_recommenders import AggregationStrategy, GRSGroupRecommender, FusionStrategy, FusionStrategyType, SaeGroupRecommender, ElsaAllInOneGroupRecommender, CombineFeaturesStrategy, CombineFeaturesStrategyType, ElsaGroupRecommender

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
    parser.add_argument('--dataset', type=str, default='EchoNest', help='Dataset to use. For now, only "LastFM1k" and "EchoNest" are supported')
    # model parameters
    parser.add_argument("--sae_run_id", type=str, default='7dcdc7663adf47f5bb9b06aca5fca746', help="Run ID of the analyzed SAE model")
    parser.add_argument("--use_base_model_from_sae", action='store_true', help="Use base model from SAE run")
    parser.add_argument("--base_run_id", type=str, default='494195a6c97f49169010f64a3bfcdf2a', help="Run ID of the base model if not using SAE base model")
    
    # Recommender parameters
    parser.add_argument("--recommender_strategy", type=str, default='LMS', help="Strategy to use for recommending. Options: 'SAE', 'ADD', ...") # TODO: Add more strategies
    parser.add_argument("--SAE_fusion_strategy", type=str, default='average', help="Only for SAE strategy. Strategy to fuse user sparse embeddings.") # TODO: Add more strategies
    parser.add_argument("--combine_features_strategy", type=str, default='none', help="Strategy to combinee features.")
    parser.add_argument("--combine_features_percentile", type=float, default=0.95, help="Aplied only if combine_features_strategy is not 'percentile'. Percentile to use for combine features.")
    parser.add_argument("--combine_features_topk", type=int, default=100, help="Aplied only if combine_features_strategy is 'topk'. combinee top_k features for each feature.")
    
    # group parameters
    parser.add_argument("--group_type", type=str, default='sim', help="Type of group to analyze. Options: 'sim', 'div', '21'")
    parser.add_argument("--group_size", type=int, default=3, help="Size of the group to analyze")
    parser.add_argument("--user_set", type=str, default='test', help="User set from which the groups where sampled (full, test, train)")
    # stable parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--target_ratio', type=float, default=0.2, help='Ratio of target interactions')
    parser.add_argument('--k', type=int, default=20, help='Evaluation at k')
    
    return parser.parse_args()

GROUP_TYPES = ['sim', 'div', '21']
RECOMMENDER_STRATEGIES = ['SAE', 'ADD', 'LMS', 'GFAR', 'EPFuzzDA', 'MPL', 'ELSA', 'ELSA_INT']
SAE_FUSION_STRATEGIES = [strategy.value for strategy in FusionStrategyType]
COMBINE_FEATURES_STRATEGIES = [strategy.value for strategy in CombineFeaturesStrategyType]
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
    
    mlflow.set_experiment(f'GR_{args.dataset}')
    mlflow.set_experiment_tags({
        'Dataset': args.dataset,
        'Task': 'Group Recommending',
        'mlflow.note.content': f'This experiment was created to evaluate group recommending for {args.dataset} dataset'
    })
    popularity = Popularity(interactions)
    
    if args.recommender_strategy == 'SAE':
        run_name = f'{args.group_type}_{args.group_size}_{args.recommender_strategy}_{args.SAE_fusion_strategy}_{args.top_k}_{args.embedding_dim}_{TIMESTAMP}'
    else:
        run_name = f'{args.group_type}_{args.group_size}_{args.recommender_strategy}_{TIMESTAMP}'
        
    def normalize_rel_scores(rel_scores):
        maxs = np.max(rel_scores, axis=-1, keepdims=True)
        mins = np.min(rel_scores, axis=-1, keepdims=True)
        rel_scores = (rel_scores - mins) / (maxs - mins)
        return rel_scores
        
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))
        ndcgs = []
        rel_scores_per_item = []
        popularities = []
        for group in tqdm(groups, desc='Group Recommending', total=len(groups)):
            group_interactions = interactions[group]
            inputs, targets = Utils.split_input_target_interactions(group_interactions, args.target_ratio, args.seed)
            inputs = torch.tensor(inputs.toarray(), device=device)
            mask = (inputs.sum(axis=0).squeeze() != 0).unsqueeze(0).repeat(inputs.shape[0], 1)
            
            group_recommendations = group_recommender.recommend_for_group(inputs, args.k, mask)
            
            user_rel_scores, user_idxs = elsa.recommend(inputs, None, mask=mask)
            additional_interactions_idx = user_idxs[:, :2000]
            rows = np.arange(additional_interactions_idx.shape[0])[:, None]  # shape (3, 1)
            targets = targets.toarray()
            targets[rows, additional_interactions_idx] = 1
            targets = torch.tensor(targets, device=device)
            
            
            # ttt = np.ones_like(user_rel_scores) / np.log2(np.arange(user_rel_scores.shape[1]) + 2)
            # min_value = 1 / np.log2(np.count_nonzero(user_rel_scores, axis=-1) + 2)
            # min_value_exp = min_value[:, np.newaxis]
            # user_rel_scores = np.where(ttt < min_value_exp, min_value_exp, ttt)
            
            sorted_user_idxs = np.argsort(user_idxs, axis=-1)
            rows = np.arange(user_idxs.shape[0])[:, None]
            user_rel_scores = normalize_rel_scores(user_rel_scores[rows, sorted_user_idxs])
            
            
            
            #TODO: vymyslet evaluacni metody
            rel_score_per_item = Utils.rel_score_per_item(group_recommendations, user_rel_scores)
            # ndcg = Utils.rel_ndcg_at_k(np.tile(group_recommendations, (3, 1)), user_rel_scores, args.k)
            
            recommendations = torch.tensor(group_recommendations, device=device).unsqueeze(0).repeat(inputs.shape[0], 1)
            ndcg = Utils.evaluate_ndcg_at_k_from_top_indices(torch.tensor(recommendations, device=device), torch.tensor(targets, device=device), args.k)
            popularities.append(popularity.popularity_score(group_recommendations))
            
            rel_scores_per_item.append(rel_score_per_item)
            ndcgs.append(ndcg)

        ndcgs_means = np.mean(ndcgs, axis=1)
        ndcgs_mins = np.min(ndcgs, axis=1)
        ndcgs_maxs = np.max(ndcgs, axis=1)
        rel_scores_per_item_means = np.mean(rel_scores_per_item, axis=1)
        rel_scores_per_item_mins = np.min(rel_scores_per_item, axis=1)
        rel_scores_per_item_maxs = np.max(rel_scores_per_item, axis=1)
        popularities_means = np.mean(popularities, axis=1)
        
        mlflow.log_metrics({
            f'NDCG{args.k}/mean': float(np.median(ndcgs_means)),
            f'NDCG{args.k}/std': float(np.std(ndcgs_means)),
            f'NDCG{args.k}/min': float(np.median(ndcgs_mins)),
            f'NDCG{args.k}/max': float(np.median(ndcgs_maxs)),
            f'NDCG{args.k}/min:mean': float(np.median(ndcgs_mins/ndcgs_means)),
            f'ScorPerItem/mean': float(np.median(rel_scores_per_item_means)),
            f'ScorPerItem/min': float(np.median(rel_scores_per_item_mins)),
            f'ScorPerItem/min:mean': float(np.median(rel_scores_per_item_mins/rel_scores_per_item_means)),
            f'Popularity/mean': float(np.median(popularities_means)),
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
        assert args.SAE_fusion_strategy in SAE_FUSION_STRATEGIES, f'Fusion strategy {args.SAE_fusion_strategy} not supported'
        assert args.combine_features_strategy in COMBINE_FEATURES_STRATEGIES, f'Combine features strategy {args.combine_features_strategy} not supported'
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
        sae_topk_aux = int(sae_params.get('topk_aux', 0))
        sae_n_batches_to_dead = int(sae_params.get('n_batches_to_dead', 0))
        sae_normalize = bool(sae_params.get('normalize', False))
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
        combine_features_strategy = CombineFeaturesStrategy.get_combine_features_strategy(CombineFeaturesStrategyType(args.combine_features_strategy), decoder=sae.decoder_w, percentile=args.combine_features_percentile, k=args.combine_features_topk)
        fusion_strategy = FusionStrategy.get_fusion_strategy(FusionStrategyType(args.SAE_fusion_strategy), k = int(args.top_k))
        group_recommender = SaeGroupRecommender(elsa, sae, fusion_strategy, combine_features_strategy)
    elif args.recommender_strategy == 'ELSA':
        group_recommender = ElsaGroupRecommender(elsa, FusionStrategy.get_fusion_strategy(FusionStrategyType('average')))
    elif args.recommender_strategy == 'ELSA_INT':
        group_recommender = ElsaAllInOneGroupRecommender(elsa)
    else: # other strategies
        args.SAE_fusion_strategy = None
        aggregator = AggregationStrategy.getAggregator(args.recommender_strategy)
        group_recommender = GRSGroupRecommender(elsa, aggregator)
    
    
    recommend(args, group_recommender, elsa, groups, interactions)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)