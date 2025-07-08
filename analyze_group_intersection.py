import argparse
import subprocess
import mlflow

GROUP_TYPES = [
    'sim',
    'outlier',
    'random',
]

def get_sae_run_ids_from_experiment(experiment_id):
    """
    Query MLflow for all SAE run IDs in the given experiment.
    """
    experiment = mlflow.get_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_id}' not found in MLflow.")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    sae_infos = [(row.run_id, row["params.note"]) for _, row in runs.iterrows()]
    return sae_infos

def main():
    parser = argparse.ArgumentParser(description='Run analyze_group_embedding_intersection.py for all SAE runs in an MLflow experiment and all group types.')
    parser.add_argument('--experiment_id', type=str, required=True, help='MLflow experiment name or ID')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., LastFM1k, EchoNest, MovieLens)')
    parser.add_argument('--user_set', type=str, default='valid', help='User set to analyze (default: valid)')
    parser.add_argument('--group_size', type=int, default=3, help='Size of the groups to analyze (default: 3)')
    args = parser.parse_args()

    sae_infos = get_sae_run_ids_from_experiment(args.experiment_id)
    if not sae_infos:
        print(f"No SAE runs found in experiment '{args.experiment_id}'.")
        return

    for acts in ['with_acts', 'without_acts']:
        for sae_run_id, note in sae_infos:
            for group_type in GROUP_TYPES:
                cmd = [
                    'python', 'analyze_group_embedding_intersection.py',
                    '--dataset', args.dataset,
                    '--sae_run_id', sae_run_id,
                    '--group_type', group_type,
                    '--group_size', str(args.group_size),
                    '--user_set', args.user_set,
                    '--note', note + '_' + acts,
                    '--topk_inference' if acts == 'with_acts' else '',
                ]
                cmd = [arg for arg in cmd if arg]
                print('Running:', ' '.join(cmd))
                subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
