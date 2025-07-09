import argparse
import logging
import pickle
import sys
import torch
from datasets import LastFm1kLoader, MovieLensLoader
from models import ELSA, BasicSAE, TopKSAE
from utils.popularity import Popularity
import mlflow
import numpy as np
import time
import os
import datetime
from tqdm import tqdm
from utils import Utils
from group_recommenders import BaseGroupRecommender
import scipy.sparse as sp
from group_recommenders import ResultsAggregationStrategy, GRSGroupRecommender, FusionStrategy, FusionStrategyType, SaeGroupRecommender, ElsaInteractionsGroupRecommender, ElsaGroupRecommender, PopularGroupRecommender

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
    parser.add_argument('--dataset', type=str, default='MovieLens', help='Dataset to use. For now, only "LastFM1k" and "MovieLens" are supported')
    # model parameters
    parser.add_argument("--sae_run_id", type=str, default='e6f874617a574800b8d5a11254711223', help="Run ID of the analyzed SAE model")
    parser.add_argument("--use_base_model_from_sae", action='store_true', help="Use base model from SAE run")
    parser.add_argument("--base_run_id", type=str, default='null', help="Run ID of the base model if not using SAE base model")
    
    # Recommender parameters
    parser.add_argument("--recommender_strategy", type=str, default='SAE', help="Strategy to use for recommending. Options: 'SAE', 'ADD', ...")
    parser.add_argument("--SAE_fusion_strategy", type=str, default='common_features', help="Only for SAE strategy. Strategy to fuse user sparse embeddings.")
    parser.add_argument("--normalize_users_embeddings", action='store_true', help="Whether to normalize user embeddings before recommending to make for them similar weight.")
    
    # configurations
    parser.add_argument('--topk_inference', action='store_true', help='Whether to use top-k activation during inference')
    
    # group parameters
    parser.add_argument("--group_type", type=str, default='sim', help="Type of group to analyze. Options: 'sim', 'outlier', '21'")
    parser.add_argument("--group_set", type=str, default='valid', help="Set of groups to analyze. Options: 'valid', 'test'")
    parser.add_argument("--group_size", type=int, default=3, help="Size of the group to analyze")
    parser.add_argument("--user_set", type=str, default='valid', help="User set from which the groups where sampled (full, test, train)")
    # stable parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--target_ratio', type=float, default=0.5, help='Ratio of target interactions from initial interactions')
    parser.add_argument('--k', type=int, default=20, help='Evaluation at k')
    parser.add_argument('--note', type=str, default='', help='Note to add to the experiment')
    
    return parser.parse_args()

GROUP_TYPES = ['sim', 'outlier', 'random']
RECOMMENDER_STRATEGIES = ['SAE', 'ADD', 'LMS', 'GFAR', 'EPFuzzDA', 'MPL', 'ELSA', 'ELSA_INT', 'POPULAR']
SAE_FUSION_STRATEGIES = [strategy.value for strategy in FusionStrategyType]
GROUP_SIZES = [3,5]

def get_groups_path(dataset, group_type, group_size, user_set):
    path = f'data/synthetic_groups/{dataset}/{user_set}/'
    if group_type == 'sim':
        filename = f'similar_{group_size}_{args.group_set}.npy'
    elif group_type == 'outlier':
        filename = f'opposing_2_1_{args.group_set}.npy'
    elif group_type == 'random':
        filename = f'random_{group_size}_{args.group_set}.npy'
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
        
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))
        all_ndcgs, all_dcgs, all_idcgs = [], [], []
        all_ndcg_com, all_dcg_com, all_idcg_com = [], [], []
        all_prec, all_rec = [], []
        all_prec_com, all_rec_com = [], []
        all_hit, all_hit_com = [], []
        total_time = 0
        all_gt_user, all_gt_group = [], []
        all_candidates = []
        popularities = []
        recommendations_similarities = []
        all_group_recommendations = []
        for group in tqdm(groups, desc='Group Recommending', total=len(groups)):
            group_interactions = interactions[group]

            inputs, targets = Utils.split_input_target_interactions_for_groups(group_interactions, args.target_ratio, args.seed)
            inputs, targets = torch.tensor(inputs, device=device, dtype=torch.float32), torch.tensor(targets, device=device, dtype=torch.float32)
            mask = (inputs.sum(axis=0).squeeze() != 0).unsqueeze(0).repeat(inputs.shape[0], 1)

            start = time.perf_counter()
            group_recommendations_raw = group_recommender.recommend_for_group(inputs, args.k, mask)
            finish = time.perf_counter()
            total_time += finish - start
            group_recommendations = torch.tensor(group_recommendations_raw, device=device).unsqueeze(0).repeat(inputs.shape[0], 1)


            common_items = torch.zeros(interactions.shape[1], device=device)
            common_items += (targets.sum(axis=0).squeeze() == args.group_size).float()
            
            gt_user = targets.sum().mean().cpu()
            gt_group = common_items.sum().cpu()
            
            candidates = (inputs.shape[1] - mask[0].sum()).cpu()

            ndcg, dcg, idcg = Utils.evaluate_ndcg_at_k_from_top_indices(group_recommendations, targets, args.k)
            ndcg_com, dcg_com, idcg_com = Utils.evaluate_ndcg_at_k_from_top_indices(group_recommendations[0].unsqueeze(0), common_items.unsqueeze(0), args.k)
            precision = Utils.evaluate_precision_at_k_from_top_indices(group_recommendations, targets, args.k)
            precision_com = Utils.evaluate_precision_at_k_from_top_indices(group_recommendations[0].unsqueeze(0), common_items.unsqueeze(0), args.k)
            recall = Utils.evaluate_recall_at_k_from_top_indices(group_recommendations, targets, args.k)
            recall_com = Utils.evaluate_recall_at_k_from_top_indices(group_recommendations[0].unsqueeze(0), common_items.unsqueeze(0), args.k)
            hit = Utils.evaluate_hitrate_at_k_from_top_indices(group_recommendations, targets, args.k)
            hit_com = Utils.evaluate_hitrate_at_k_from_top_indices(group_recommendations[0].unsqueeze(0), common_items.unsqueeze(0), args.k)
            popularity_score = popularity.popularity_score(group_recommendations[0].cpu().numpy())
            recommendations_similarity = Utils.recommendations_similarity(group_recommendations[0], elsa)
            
            # generate more info for test set
            if args.group_set == 'test':
                all_recs = group_recommendations_raw
                all_group_recommendations.append(all_recs)
            
            all_ndcgs.append(ndcg)
            all_dcgs.append(dcg)
            all_idcgs.append(idcg)
            
            all_ndcg_com.append(ndcg_com)
            all_dcg_com.append(dcg_com)
            all_idcg_com.append(idcg_com)
            
            all_prec.append(precision)
            all_rec.append(recall)
            
            all_prec_com.append(precision_com)
            all_rec_com.append(recall_com)
            
            all_hit.append(hit)
            all_hit_com.append(hit_com)
            
            all_gt_group.append(gt_group)
            all_gt_user.append(gt_user)
            
            all_candidates.append(candidates)
            
            popularities.append(popularity_score)
            recommendations_similarities.append(recommendations_similarity)
            
        ndcgs_means = np.mean(all_ndcgs, axis=1)        
        
        ndcgs_mins = np.min(all_ndcgs, axis=1)
        dcgs_mins = np.min(all_dcgs, axis=1)
        idcgs_mins = np.min(all_idcgs, axis=1)
        recall_mins = np.min(all_rec, axis=1)
        prec_mins = np.min(all_prec, axis=1)
        hit_mins = np.min(all_hit, axis=1)
        
        popularities_means = np.mean(popularities, axis=1)
        
        logs = {
            'Group_NDCG': np.array(all_ndcg_com),
            'User_NDCG_Means': np.array(ndcgs_means),
            'User_NDCG_Mins': np.array(ndcgs_mins),
            "Popularity": np.array(popularities_means),
            "Recommendations": np.array(all_group_recommendations),
        }
        temp_path = './logs.pkl'
        with open(temp_path, 'wb') as f:
            pickle.dump(logs, f)
        mlflow.log_artifact(temp_path)
        os.remove(temp_path)
        logging.info('Logs successfully saved')
        
        mlflow.log_metrics({
            f'NDCG{args.k}_com': float(np.mean(all_ndcg_com)),
            f'DCG{args.k}_com': float(np.mean(all_dcg_com)),
            f'IDCG{args.k}_com': float(np.mean(all_idcg_com)),
            
            f'NDCG{args.k}_min': float(np.mean(ndcgs_mins)),
            f'DCG{args.k}_min': float(np.mean(dcgs_mins)),
            f'IDCG{args.k}_min': float(np.mean(idcgs_mins)),
            
            f'P{args.k}_min': float(np.mean(prec_mins)),
            f'P{args.k}_com': float(np.mean(all_prec_com)),
            
            f'R{args.k}_com': float(np.mean(all_rec_com)),
            f'R{args.k}_min': float(np.mean(recall_mins)),
            
            f'Hit{args.k}_com': float(np.mean(all_hit_com)),
            f'Hit{args.k}_min': float(np.mean(hit_mins)),
            
            f'GT_group': float(np.mean(all_gt_group)),
            f'GT_user': float(np.mean(all_gt_user)),
            
            f'Candidates': float(np.mean(all_candidates)),
            
            f'NDCG{args.k}_avg': float(np.mean(ndcgs_means)),
            f'NDCG{args.k}/min:mean': float(np.mean(np.divide(ndcgs_mins, ndcgs_means, out=np.zeros_like(ndcgs_mins), where=ndcgs_means!=0))),
            f'Popularity/mean': float(np.mean(popularities_means)),
            f'RecommendationsSimilarity/mean': float(np.mean(recommendations_similarities)),
            f'Time/mean': float(total_time / len(groups)),
        })
        

def main(args):
    assert args.group_size in GROUP_SIZES, 'Only group size 3 is supported for now'
    assert args.group_type in ['sim', 'outlier', 'random'], 'Group type not supported'
    assert args.recommender_strategy in RECOMMENDER_STRATEGIES, 'Recommender strategy not supported'
    if not args.recommender_strategy == 'SAE' or not args.use_base_model_from_sae:
        assert args.base_run_id is not None, 'Base model run ID is required'
    
    # Load dataset
    if args.dataset == 'LastFM1k':
        dataset_loader = LastFm1kLoader()
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
        fusion_strategy = FusionStrategy.get_fusion_strategy(FusionStrategyType(args.SAE_fusion_strategy), k = int(args.top_k))
        group_recommender = SaeGroupRecommender(elsa, sae, fusion_strategy, normalize_user_embedding=args.normalize_users_embeddings)
    elif args.recommender_strategy == 'ELSA':
        group_recommender = ElsaGroupRecommender(elsa, FusionStrategy.get_fusion_strategy(FusionStrategyType('average')))
    elif args.recommender_strategy == 'POPULAR':
        group_recommender = PopularGroupRecommender(interactions)
    elif args.recommender_strategy == 'ELSA_INT':
        group_recommender = ElsaInteractionsGroupRecommender(elsa)
    else: # other strategies
        args.SAE_fusion_strategy = None
        aggregator = ResultsAggregationStrategy.getAggregator(args.recommender_strategy)
        group_recommender = GRSGroupRecommender(elsa, aggregator)
    
    
    recommend(args, group_recommender, elsa, groups, interactions)
    

if __name__ == '__main__':
    args = parse_arguments()
    main(args)