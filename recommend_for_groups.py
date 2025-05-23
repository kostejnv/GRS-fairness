import argparse
import logging
import sys
import torch
from datasets import EchoNestLoader, LastFm1kLoader, DataLoader, MovieLensLoader
from models import ELSA, ELSAWithSAE, BasicSAE, TopKSAE, SAE
from popularity import Popularity
import mlflow
import numpy as np
import time
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
    parser.add_argument('--dataset', type=str, default='LastFM1k', help='Dataset to use. For now, only "LastFM1k" and "EchoNest" and "MovieLens" are supported')
    # model parameters
    parser.add_argument("--sae_run_id", type=str, default='e6f874617a574800b8d5a11254711223', help="Run ID of the analyzed SAE model")
    parser.add_argument("--use_base_model_from_sae", action='store_true', help="Use base model from SAE run")
    parser.add_argument("--base_run_id", type=str, default='null', help="Run ID of the base model if not using SAE base model")
    
    # Recommender parameters
    parser.add_argument("--recommender_strategy", type=str, default='SAE', help="Strategy to use for recommending. Options: 'SAE', 'ADD', ...") # TODO: Add more strategies
    parser.add_argument("--SAE_fusion_strategy", type=str, default='common_features', help="Only for SAE strategy. Strategy to fuse user sparse embeddings.") # TODO: Add more strategies
    parser.add_argument("--combine_features_strategy", type=str, default='none', help="Strategy to combinee features.")
    parser.add_argument("--combine_features_percentile", type=float, default=0.50, help="Aplied only if combine_features_strategy is not 'percentile'. Percentile to use for combine features.")
    parser.add_argument("--combine_features_topk", type=int, default=100, help="Aplied only if combine_features_strategy is 'topk'. combinee top_k features for each feature.")
    
    # group parameters
    parser.add_argument("--group_type", type=str, default='sim', help="Type of group to analyze. Options: 'sim', 'div', '21'")
    parser.add_argument("--group_size", type=int, default=3, help="Size of the group to analyze")
    parser.add_argument("--user_set", type=str, default='train', help="User set from which the groups where sampled (full, test, train)")
    # stable parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--target_ratio', type=float, default=0.5, help='Ratio of target interactions from initial interactions')
    parser.add_argument('--add_interactions', type=int, default=0, help='Number of additional interactions that should be added for each user (used if we have low number of interactions). Note: The final number of target interactions will be target_ratio * interactions + add_interactions')
    parser.add_argument('--k', type=int, default=20, help='Evaluation at k')
    parser.add_argument('--note', type=str, default='', help='Note to add to the experiment')
    parser.add_argument('--topk_inference', action='store_true', help='Whether to use top-k activation during inference')
    
    return parser.parse_args()

GROUP_TYPES = ['sim', 'div', '21', 'random']
RECOMMENDER_STRATEGIES = ['SAE', 'ADD', 'LMS', 'GFAR', 'EPFuzzDA', 'MPL', 'ELSA', 'ELSA_INT']
SAE_FUSION_STRATEGIES = [strategy.value for strategy in FusionStrategyType]
COMBINE_FEATURES_STRATEGIES = [strategy.value for strategy in CombineFeaturesStrategyType]
GROUP_SIZES = [3,5]

def get_groups_path(dataset, group_type, group_size, user_set):
    path = f'data/synthetic_groups/{dataset}/{user_set}/'
    if group_type == 'sim':
        filename = f'similar_{group_size}.npy'
    elif group_type == 'div':
        filename = f'divergent_{group_size}.npy'
    elif group_type == '21':
        filename = f'opposing_2_1.npy'
    elif group_type == 'random':
        filename = f'random_{group_size}.npy'
    return path + filename
    
def recommend(args, group_recommender: BaseGroupRecommender, elsa: ELSA, groups, interactions: sp.csr_matrix):
    logging.info('Recommending for groups')
    
    add_interactions = args.add_interactions
    
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
        
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))
        ndcgs = []
        common_items_ndcgs = []
        total_time = 0
        rel_scores_per_item = []
        ranks_per_item = []
        popularities = []
        recommendations_similarities = []
        all_group_recommendations = []
        for group in tqdm(groups, desc='Group Recommending', total=len(groups)):
            group_interactions = interactions[group]
            interactions_count = group_interactions.sum(axis=1).A1
            
            
            # add interactions
            if add_interactions > 0:
                ints = torch.tensor(group_interactions.toarray(), device=device)
                _, rec_items = elsa.recommend(ints, add_interactions, mask=None)
                additional_interactions_idx = rec_items[:, :add_interactions]
                
                # Add new interactions to group_interactions
                additional_data = np.ones_like(additional_interactions_idx.flatten())
                additional_rows = np.repeat(np.arange(group_interactions.shape[0]), add_interactions)
                additional_cols = additional_interactions_idx.flatten()
                additional_matrix = sp.csr_matrix(
                    (additional_data, (additional_rows, additional_cols)),
                    shape=group_interactions.shape
                )
                group_interactions += additional_matrix
                
                
            
            final_target_ratio = np.mean((interactions_count * args.target_ratio + add_interactions) / (interactions_count + add_interactions))
            inputs, targets = Utils.split_input_target_interactions_for_groups(group_interactions, final_target_ratio, args.seed)
            # targets += additional_matrix
            inputs, targets = torch.tensor(inputs, device=device, dtype=torch.float32), torch.tensor(targets, device=device, dtype=torch.float32)
            mask = (inputs.sum(axis=0).squeeze() != 0).unsqueeze(0).repeat(inputs.shape[0], 1)
            
            start = time.perf_counter()
            group_recommendations = group_recommender.recommend_for_group(inputs, None, mask)
            finish = time.perf_counter()
            all_group_recommendations.append(group_recommendations)
            group_recommendations = group_recommendations[:, :args.k]
            total_time += finish - start
            group_recommendations = torch.tensor(group_recommendations, device=device).unsqueeze(0).repeat(inputs.shape[0], 1)
            elsa_scores = elsa.decode(elsa.encode(inputs)) - inputs
            elsa_scores = elsa.normalize_relevance_scores(elsa_scores)
            elsa_scores = torch.where(mask, -torch.inf, elsa_scores)
            
            # # get top 20 common items
            # common_items_count = 20
            common_items = torch.zeros(interactions.shape[1], device=device)
            # at first from targets
            common_items += (targets.sum(axis=0).squeeze() == args.group_size).float()
            # # then from recommendations
            # need_to_add = int((common_items_count - common_items.sum()).cpu().item())
            # if need_to_add > 0:
            #     min_scores = elsa_scores.min(dim=0)[0]
            #     top_scores = torch.topk(min_scores, need_to_add).indices
            #     common_items += torch.zeros_like(common_items).scatter_(0, top_scores, 1)
            
            
            rank_per_item = Utils.ranks_per_item(group_recommendations[0], elsa_scores.detach().cpu().numpy())
            
            
            ndcg = Utils.evaluate_ndcg_at_k_from_top_indices(group_recommendations, targets, args.k)
            common_items_ndcg = Utils.evaluate_ndcg_at_k_from_top_indices(group_recommendations[0].unsqueeze(0), common_items.unsqueeze(0), args.k)
            popularity_score = popularity.popularity_score(group_recommendations[0].cpu().numpy())
            recommendations_similarity = Utils.recommendations_similarity(group_recommendations[0], elsa)
            
            
            ndcgs.append(ndcg)
            common_items_ndcgs.append(common_items_ndcg)
            popularities.append(popularity_score)
            ranks_per_item.append(rank_per_item)
            recommendations_similarities.append(recommendations_similarity)
            
        ndcgs_means = np.mean(ndcgs, axis=1)
        ndcgs_mins = np.min(ndcgs, axis=1)
        ndcgs_maxs = np.max(ndcgs, axis=1)
        ranks_per_item_means = np.mean(ranks_per_item, axis=1)
        ranks_per_item_mins = np.min(ranks_per_item, axis=1)
        ranks_per_item_maxs = np.max(ranks_per_item, axis=1)
        popularities_means = np.mean(popularities, axis=1)
        
        temp_path = './recommendations.npy'
        np.save(temp_path, np.array(all_group_recommendations))
        mlflow.log_artifact(temp_path)
        os.remove(temp_path)
        logging.info('Recommendations successfully saved')
        
        mlflow.log_metrics({
            f'CommonItemsNDCG{args.k}/median': float(np.median(common_items_ndcg)),
            f'NDCG{args.k}/mean': float(np.median(ndcgs_means)),
            f'NDCG{args.k}/std': float(np.std(ndcgs_means)),
            f'NDCG{args.k}/min': float(np.median(ndcgs_mins)),
            f'NDCG{args.k}/max': float(np.median(ndcgs_maxs)),
            f'NDCG{args.k}/min:mean': float(np.median(np.divide(ndcgs_mins, ndcgs_means, out=np.zeros_like(ndcgs_mins), where=ndcgs_means!=0))),
            f'RanksPerItem/mean': float(np.median(ranks_per_item_means)),
            f'RanksPerItem/min': float(np.median(ranks_per_item_mins)),
            f'RanksPerItem/min:mean': float(np.median(np.divide(ranks_per_item_mins, ranks_per_item_means, out=np.zeros_like(ranks_per_item_mins), where=ranks_per_item_means!=0))),
            f'Popularity/mean': float(np.median(popularities_means)),
            f'RecommendationsSimilarity/mean': float(np.median(recommendations_similarities)),
            f'Time/mean': float(total_time / len(groups)),
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
    elif args.dataset == 'MovieLens':
        dataset_loader = MovieLensLoader()
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
        args.reconstruction_loss = sae_params['reconstruction_loss']
        args.normalize = sae_params.get('normalize', 'False')
        args.contrastive_coef = float(sae_params.get('contrastive_coef', 0))
    
        sae_artifact_path = sae_run.info.artifact_uri
        sae_artifact_path = './' + sae_artifact_path[sae_artifact_path.find('mlruns'):]
        
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
            "topk_inference": args.topk_inference,
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