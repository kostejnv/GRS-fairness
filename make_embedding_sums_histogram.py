import torch
import random
from datasets import EchoNestLoader, LastFm1kLoader, MovieLensLoader, DataLoader
from models import ELSA, BasicSAE, TopKSAE
from utils import Utils
import matplotlib.pyplot as plt
import mlflow

# Hardcoded parameters for the specific use case
DATASET = 'MovieLens'
SAE_RUN_ID = 'bbae35b675674be99f7ec013dc869b35'
USER_SET = 'full'
USER_SAMPLE = 5000

device = Utils.set_device()

sae_run = mlflow.get_run(SAE_RUN_ID)
sae_params = sae_run.data.params

embedding_dim = int(sae_params['embedding_dim'])
top_k = sae_params.get('top_k', None)
sae_model = sae_params['model']

sae_artifact_path = sae_run.info.artifact_uri
if sae_artifact_path is not None and 'mlruns' in sae_artifact_path:
    sae_artifact_path = './' + sae_artifact_path[sae_artifact_path.find('mlruns'):]
else:
    raise RuntimeError('Could not determine SAE artifact path')

base_model_run = mlflow.get_run(sae_params['base_run_id'])
base_model_params = base_model_run.data.params
base_artifact_path = base_model_run.info.artifact_uri
if base_artifact_path is not None and 'mlruns' in base_artifact_path:
    base_artifact_path = './' + base_artifact_path[base_artifact_path.find('mlruns'):]
else:
    raise RuntimeError('Could not determine base model artifact path')

# Only MovieLens supported for this script
if DATASET == 'MovieLens':
    dataset_loader = MovieLensLoader()
else:
    raise ValueError(f'Dataset {DATASET} not supported.')

dataset_loader.prepare({'val_ratio': 0.1, 'test_ratio': 0.1, 'seed': 42})
if USER_SET == 'full':
    csr_interactions = dataset_loader.csr_interactions
elif USER_SET == 'train':
    csr_interactions = dataset_loader.train_csr
elif USER_SET == 'test':
    csr_interactions = dataset_loader.test_csr
else:
    raise ValueError(f'User set {USER_SET} not supported.')

if csr_interactions is None:
    raise RuntimeError('csr_interactions is None. Check if the dataset was loaded and processed correctly.')

if USER_SAMPLE > 0:
    sample_users = random.sample(range(csr_interactions.shape[0]), min(USER_SAMPLE, csr_interactions.shape[0]))
    csr_interactions = csr_interactions[sample_users, :]

interactions_batches = DataLoader(csr_interactions, batch_size=1024, device=device, shuffle=False)

base_factors = int(base_model_params['factors'])
base_items = int(base_model_params['items'])
sae_embedding_dim = int(sae_params['embedding_dim'])
sae_top_k = int(sae_params['top_k'])

cfg = {
    'input_dim': base_factors,
    'embedding_dim': sae_embedding_dim,
    'k': sae_top_k,
    'device': device,
}

elsa = ELSA(base_items, base_factors)
optimizer = torch.optim.Adam(elsa.parameters())
Utils.load_checkpoint(elsa, optimizer, f'{base_artifact_path}/checkpoint.ckpt', device)
elsa.to(device)
elsa.eval()

if sae_params['model'] == 'BasicSAE':
    sae = BasicSAE(base_factors, sae_embedding_dim, cfg).to(device)
elif sae_params['model'] == 'TopKSAE':
    sae = TopKSAE(base_factors, sae_embedding_dim, cfg).to(device)
else:
    raise ValueError(f'Model {sae_params["model"]} not supported.')

sae_optimizer = torch.optim.Adam(sae.parameters())
Utils.load_checkpoint(sae, sae_optimizer, f'{sae_artifact_path}/checkpoint.ckpt', device)
sae.to(device)
sae.eval()

batches_embeddings = []
for batch in interactions_batches:
    batch_embeddings = sae.encode(elsa.encode(batch))[0]
    batches_embeddings.append(batch_embeddings.detach().to('cpu'))
user_embeddings_with_zeros = torch.cat(batches_embeddings)

embeddings_sums = torch.sum(user_embeddings_with_zeros, dim=1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(embeddings_sums.to('cpu').detach().numpy(), bins=30)
ax.set_title('Embedding Sums Histogram')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
plt.tight_layout()
fig.savefig('embedding_sums_histogram.pdf')
plt.close()
