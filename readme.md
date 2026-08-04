# Zero-Day Fallacies: Reproducible NIDS Experiments

This repository accompanies an anonymous AISec submission on evaluating network intrusion detection systems (NIDS) under **leave-one-class-out (LOCO)** or **zero-day** conditions.

The code supports:

- closed-set and leave-one-attack-class-out evaluation;
- flow-based and heterogeneous host–flow graph models;
- clean and Gaussian-noise-augmented training;
- experiments on **CIDDS-001** and **NF-UNSW-NB15-v3**.

> **Important interpretation.** In a LOCO run, the selected attack class is removed from training and validation but retained in the test set. A held-out sample is considered detected when the model assigns it to any non-benign class. The experiment therefore measures whether unseen attacks are rejected as malicious, not whether their unavailable attack-family label is predicted.

---

## 1. Repository structure

The main entry point is `main.py`:

```text
.
├── main.py                 # Parses the configuration and starts an experiment
├── config.py               # Command-line arguments and defaults
├── data/
│   ├── download.py         # CIDDS-001 download helper
│   ├── get_dataset.py      # Dataset selection and cached split loading
│   ├── loader.py           # Host–flow graph construction and data loaders
│   └── preprocess.py       # CIDDS and NF-UNSW preprocessing
├── models/                 # MLP, LSTM, Transformer, and heterogeneous GNNs
├── training/
│   ├── train.py            # Baselines and individual encoders
│   ├── train_master.py     # BRIDGE/meta-model training
│   ├── test.py             # Evaluation routines
│   └── mahalanobis.py      # Mahalanobis scoring
├── attacks/                # Optional input perturbation code
├── utils/                  # Metrics, reports, plots, and class weighting
├── models/weights/         # Saved checkpoints
├── results/dataframes/     # Per-sample predictions and scores
└── figures/                # Generated plots
```

`main.py` dispatches to the standard training pipeline unless `master_model` is enabled, in which case it runs the BRIDGE/meta-model pipeline.

---

## 2. Environment setup

Create an isolated Python environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell
```

Install the repository requirements when a lock file or `requirements.txt` is provided:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The code directly depends on the following packages:

```bash
pip install numpy pandas scipy scikit-learn pyarrow joblib tqdm matplotlib seaborn
```

Install PyTorch and PyTorch Geometric using versions compatible with your CUDA installation. See the official PyTorch and PyG installation instructions rather than mixing arbitrary CUDA wheels.

Create the output directories before the first run:

```bash
mkdir -p models/weights results/dataframes figures
```

The device is selected automatically:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

A CUDA-capable GPU is strongly recommended for the full graph experiments.

---

## 3. Inspect the command-line interface

Run:

```bash
python main.py --help
```

The experiment code uses the following configuration fields:

| Argument | Purpose |
|---|---|
| `dataset` | `CIDDS` or `NF-UNSW-NB15-v3` |
| `model` | Base architecture, e.g. `HeteroSage`, `HeteroGAT`, `HGT`, `MLP`, `LSTM`, `Transformer`, or `SVM` |
| `unknown_classes` | One or more classes excluded from training and validation |
| `epochs` | Number of training epochs |
| `lr` | Adam learning rate |
| `hidden_dim` | Hidden/embedding dimension |
| `out_dim` | Number of known classes in the current run |
| `num_layers` | Number of model layers |
| `batch_size` | Number of 1,024-flow subgraphs per batch |
| `num_runs` | Number of repetitions stored in the output parquet file |
| `class_weight` | `balanced` or comma-separated manual weights |
| `pre_transform` | Apply graph transforms and Gaussian noise with standard deviation `0.05` |
| `mahalanobis` | Enable Mahalanobis-distance scoring |
| `perturbed` | Apply the optional perturbation routine to the training loader |
| `master_model` | Run the BRIDGE/meta-model pipeline |
| `master_arch` | `MLP`, `GatedMLP`, or `ExtendedGatedMLP` |
| `path` | Checkpoint for the noise-trained encoder |
| `path_raw` | Checkpoint for the clean/raw encoder |

The examples below use flag-style booleans, such as `--pre_transform`. If the local `config.py` uses explicit boolean values, use `--pre_transform True`, `--master_model True`, and so on, as shown by `python main.py --help`.

---

## 4. Dataset preparation

### 4.1 CIDDS-001

CIDDS-001 is downloaded automatically when `--dataset CIDDS` is used and `data/CIDDS/` is empty. The downloader retrieves the official archive and extracts it under:

```text
data/CIDDS/CIDDS-001/
```

The preprocessing code uses:

```text
data/CIDDS/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv
```

It creates stratified 80/10/10 train, validation, and test splits:

```text
data/CIDDS/df_train.parquet
data/CIDDS/df_val.parquet
data/CIDDS/df_test.parquet
```

CIDDS class names are case-sensitive:

```text
benign
bruteForce
dos
pingScan
portScan
```

Note that the code uses `dos` in lowercase.

To regenerate the splits, remove the three cached parquet files and rerun an experiment.

### 4.2 NF-UNSW-NB15-v3

Place the raw parquet file at exactly:

```text
data/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.parquet
```

Then run preprocessing explicitly:

```bash
python -m data.preprocess
```

The current preprocessing entry point calls the compact **no-port/no-L7** variant and writes:

```text
data/NF-UNSW-NB15-v3-compact/df_train.parquet
data/NF-UNSW-NB15-v3-compact/df_val.parquet
data/NF-UNSW-NB15-v3-compact/df_test.parquet
data/NF-UNSW-NB15-v3-compact/attack_labels.json
```

This explicit preprocessing step is important because dataset loading reads `attack_labels.json` before starting an experiment.

Inspect the available, case-sensitive attack names with:

```bash
cat data/NF-UNSW-NB15-v3-compact/attack_labels.json
```

---

## 5. How graphs are constructed

Every flow becomes a `flow` node. Its source and destination IP addresses become or identify `host` nodes. Each flow is connected in both directions to its source and destination hosts:

```text
host → flow
flow → host
```

The loader partitions each split into consecutive subgraphs of **1,024 flows**. Therefore:

- `batch_size` is the number of subgraphs, not the number of individual flows;
- each loader item contains up to 1,024 flow nodes and its incident host nodes; and
- a trailing remainder smaller than 1,024 flows is currently omitted.

When `pre_transform` is enabled, the pipeline applies:

1. removal of isolated nodes;
2. self-loops;
3. feature normalization; and
4. Gaussian feature noise with `std = 0.05`.

---

## 6. Quick smoke test

A short CIDDS run verifies the installation and data pipeline:

```bash
python main.py \
  --dataset CIDDS \
  --model HeteroSage \
  --unknown_classes bruteForce \
  --epochs 1 \
  --lr 0.001 \
  --hidden_dim 64 \
  --out_dim 4 \
  --num_layers 2 \
  --batch_size 1 \
  --num_runs 1 \
  --class_weight balanced
```

This is only a functional test, not a paper-quality experiment.

---

## 7. Closed-set experiment

Omit `unknown_classes` to train and test on all classes:

```bash
python main.py \
  --dataset CIDDS \
  --model HeteroSage \
  --epochs 100 \
  --lr 0.001 \
  --hidden_dim 64 \
  --out_dim 5 \
  --num_layers 2 \
  --batch_size 32 \
  --num_runs 10 \
  --class_weight balanced
```

For the normal base-training pipeline, `out_dim` is corrected automatically when it does not match the number of known labels. It is still better to set it explicitly for readable experiment records.

---

## 8. Leave-one-class-out zero-day experiments

For CIDDS, run one experiment per attack class:

```bash
for unknown in bruteForce dos pingScan portScan; do
  python main.py \
    --dataset CIDDS \
    --model HeteroSage \
    --unknown_classes "$unknown" \
    --epochs 100 \
    --lr 0.001 \
    --hidden_dim 64 \
    --out_dim 4 \
    --num_layers 2 \
    --batch_size 32 \
    --num_runs 10 \
    --class_weight balanced
done
```

For a held-out class `c`:

- `c` is removed from the training and validation label columns;
- the model is trained on benign traffic and the remaining known attacks;
- `c` remains in the test labels; and
- zero-day detection counts a held-out sample as detected if it is predicted as any non-benign known attack.

To run another architecture, replace `HeteroSage` with one of the supported model names. For example:

```bash
--model HeteroGAT
--model HGT
--model MLP
--model LSTM
--model Transformer
--model SVM
```

`Transformer` is supported in the standard training pipeline but is not currently selectable as a BRIDGE encoder in `train_master.py`.

---

## 9. Clean versus noise-augmented encoders

A clean encoder is trained without `pre_transform`:

```bash
python main.py \
  --dataset CIDDS \
  --model HeteroSage \
  --unknown_classes pingScan \
  --epochs 100 \
  --lr 0.001 \
  --hidden_dim 64 \
  --out_dim 4 \
  --num_layers 2 \
  --batch_size 32 \
  --num_runs 1 \
  --class_weight balanced
```

Train the corresponding noise-augmented encoder with the same configuration plus `pre_transform`:

```bash
python main.py \
  --dataset CIDDS \
  --model HeteroSage \
  --unknown_classes pingScan \
  --epochs 100 \
  --lr 0.001 \
  --hidden_dim 64 \
  --out_dim 4 \
  --num_layers 2 \
  --batch_size 32 \
  --num_runs 1 \
  --class_weight balanced \
  --pre_transform
```

Checkpoints are saved under `models/weights/`. Their generated filenames encode the dataset, model, optimization settings, held-out classes, class weighting, and whether the run was raw or noise-transformed.

---

## 10. Mahalanobis baseline

For compatible neural models exposing `get_flow_embeddings`, enable Mahalanobis scoring:

```bash
python main.py \
  --dataset CIDDS \
  --model HeteroSage \
  --unknown_classes dos \
  --epochs 100 \
  --lr 0.001 \
  --hidden_dim 64 \
  --out_dim 4 \
  --num_layers 2 \
  --batch_size 32 \
  --num_runs 10 \
  --class_weight balanced \
  --mahalanobis
```
The covariance model is fitted on benign training embeddings. The threshold is the 95th percentile of benign training distances.

---


