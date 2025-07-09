# SAGEA: Sparse Autoencoder-based Group Embeddings Aggregation

**SAGEA** is a method for aggregating group embeddings using sparse autoencoders. This repository provides the official implementation used in the LBR paper _"SAGEA: Sparse Autoencoder-based Group Embeddings Aggregation"_ and the diploma thesis. _"Multiobjective models for group recommender systems"_. It serves as a reference for reproducing the results and benchmarking against other models.

---

## Branches

The code for both the paper and the diploma thesis is similar, but the branches are separated for clarity:

- **master**: Contains all materials related to the paper and the diploma thesis.
- **paper**: Contains only the code and analysis (results directory) related to the paper.
- **thesis**: Contains only the code and notebooks (notebooks directory) related to the diploma thesis.

## Requirements

This project requires **Python 3.10** or higher. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Reproducing the Results

To reproduce the results or evaluate the SAGEA method, follow the steps below using the scripts provided in the `scripts/` directory:

1. **Download the dataset:**
   ```bash
   ./scripts/download_*_dataset.sh
   ```

2. **Train the base model (ELSA):**
   ```bash
   ./scripts/train_elsa.sh
   ```
   _Note: Please edit the script to adjust the training parameters as needed._

3. **Train the sparse autoencoder (SAE) variants:**
   ```bash
   ./scripts/train_sae.sh
   ```
   _Note: Modify the parameters inside the script before running._

4. **Generate synthetic evaluation groups:**
   ```bash
   ./scripts/generate_groups.sh
   ```

5. **Evaluate SAGEA on generated groups:**
   ```bash
   ./scripts/recommend_sagea.sh
   ```

6. **Evaluate baseline models:**
   ```bash
   ./scripts/recommend_other_models.sh
   ```

---

## Important Note

This repository is tightly integrated with the **MLflow** framework for experiment tracking. Before running experiments, we recommend reviewing the MLflow documentation: [https://mlflow.org/classical-ml](https://mlflow.org/classical-ml)

---