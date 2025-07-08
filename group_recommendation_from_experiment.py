import argparse
import subprocess
import mlflow

FUSION_STRATEGIES = [
    'at_least_2_common_features',
    'average',
    'square_average',
    'topk',
    'max',
    'common_features',
    'wcom',
]

def get_sae_run_ids_from_experiment(experiment_id, note=None):
    experiment = mlflow.get_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_id}' not found in MLflow.")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"params.note = '{note}'" if note else "")
    sae_infos = [(row.run_id, row["params.note"], row["params.dataset"]) for _, row in runs.iterrows()]
    return sae_infos

def main():
    parser = argparse.ArgumentParser(description='Run recommend_for_groups.py for all SAE runs in an MLflow experiment.')
    parser.add_argument('--experiment_id', type=str, required=True, help='MLflow experiment name or ID')
    parser.add_argument('--note', type=str, required=True, help='Note to add to each run')
    parser.add_argument('--group_type', type=str, default='sim', help='Group type (default: sim)')
    parser.add_argument('--group_size', type=int, default=3, help='Group size (default: 3)')
    parser.add_argument('--fusion_strategy', type=str, choices=FUSION_STRATEGIES, default='average',
                        help='Fusion strategy to use (default: average)')
    parser.add_argument('--normalize_users_embedding', action='store_true',
                        help='Normalize user embedding before fusion (default: False)')
    parser.add_argument('--topk_inference', action='store_true',
                        help='Use top-k inference for group recommendations (default: False)')
    parser.add_argument('--true_note', type=str, default='',
                        help='True note to use for the runs (default: empty string)')
    args = parser.parse_args()

    sae_infos = get_sae_run_ids_from_experiment(args.experiment_id, note=args.note)
    if not sae_infos:
        print(f"No SAE runs found in experiment '{args.experiment_id}'.")
        return

    for sae_run_id, _, dataset in sae_infos:
        user_set = 'valid'
        note = args.true_note if args.true_note else args.note
        cmd = [
            'python', 'recommend_for_groups.py',
            '--dataset', dataset,
            '--sae_run_id', sae_run_id,
            '--use_base_model_from_sae',
            '--recommender_strategy', 'SAE',
            '--SAE_fusion_strategy', args.fusion_strategy,
            '--group_type', args.group_type,
            '--group_size', str(args.group_size),
            '--user_set', user_set,
            '--note', note + ('_with_acts' if args.topk_inference else '_without_acts') + ('_normalized' if args.normalize_users_embedding else ''),
        ]
        if args.normalize_users_embedding:
            cmd.append('--normalize_users_embedding')
        if args.topk_inference:
            cmd.append('--topk_inference')
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
