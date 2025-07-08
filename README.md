# SAGEA: Sparse Autoencoder-based Group Embeddings Aggregation

**SAGEA** is a method for aggregating group embeddings using sparse autoencoders. This repository provides the official implementation used in the paper _"SAGEA: Sparse Autoencoder-based Group Embeddings Aggregation"_. It serves as a reference for reproducing the results and benchmarking against other models.

---

## Requirements

This project requires **Python 3.10** or higher. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Results

The results of the experiments from the paper are available in the `results/` directory. This includes:

- CSV files containing the raw outputs of the experiments
- A Jupyter notebook (`results.ipynb`) for generating the figures and tables presented in the paper

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