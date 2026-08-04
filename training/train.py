import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import Compose, RemoveIsolatedNodes, AddSelfLoops, NormalizeFeatures
from tqdm import tqdm

from data.preprocess import preprocess_CIDDS
from data.loader import create_dataloader, GaussianNoise, build_bipartite_adjacency, create_full_graph
from data.get_dataset import get_dataset

from models.hetero_gnn import HeteroGNN, HeteroGAT, HGT
from models.mlp import MLP
from models.lstm import LSTMClassifier
from models.transformer import TransformerClassifier

from training.test import test, test_ocsvm, test_mahalanobis
from training.mahalanobis import mahalanobis_distance

from attacks.random_attack import apply_perturbation

from utils.results import save_results, cls_report, test_report

import os
import pandas as pd
from tqdm import trange
from utils.utils import make_class_weight, get_model_size

from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline

from sklearn.covariance import EmpiricalCovariance

import time
import random


def extract_flow_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []

    for batch in loader:
        batch.to(device)
        emb = model.get_flow_embeddings(batch)
        label = batch['flow'].y.argmax(dim=1)
        embeddings.append(emb.detach().cpu())
        labels.append(label.detach().cpu())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, labels


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train, df_val, df_test, features_host, features_flow, labels = get_dataset(args)

    if args.unknown_classes is not None:
        known_labels = [item for item in labels if item not in args.unknown_classes]
        df_train = df_train.drop(columns=args.unknown_classes)
        df_val = df_val.drop(columns=args.unknown_classes)
        print(f"Known labels are: {known_labels} \n Unknown labels are: {args.unknown_classes}")

        new_order_labels = known_labels + args.unknown_classes
        print(f"Appended known labels{new_order_labels}")
    else:
        known_labels = labels
        new_order_labels = known_labels
        df_train = df_train
        df_val = df_val
        print(f"Known labels are: {new_order_labels}")

    num_classes = len(known_labels)
    print("Num classes (known_labels):", num_classes)

    if args.out_dim != num_classes:
        print(f"⚠️ out_dim mismatch: args.out_dim={args.out_dim}, num_classes={num_classes}. Fixing.")
        args.out_dim = num_classes

    Y = df_train[known_labels].astype(int).values
    row_sum = Y.sum(axis=1)
    print("one-hot row_sum min/max:", row_sum.min(), row_sum.max())
    print("pct row_sum != 1:", (row_sum != 1).mean())

    if args.pre_transform:
        # args.noise = 0.05
        pre_transform = Compose([
            RemoveIsolatedNodes(),  # remove unconnected nodes
            AddSelfLoops(),  # ensure self-messages
            NormalizeFeatures(),  # normalize each node type's features
            GaussianNoise(std=args.noise)  # add small noise to features
        ])
        print(f'Noise is {args.noise}')
    else:
        pre_transform = None

    print("📤 Creating data loaders...")
    train_loader = create_dataloader(df_train, known_labels, args.batch_size, features_flow, features_host,
                                     pre_transform=pre_transform)
    val_loader = create_dataloader(df_val, known_labels, args.batch_size, features_flow, features_host,
                                   pre_transform=pre_transform)
    test_loader = create_dataloader(df_test, new_order_labels, args.batch_size, features_flow, features_host,
                                    pre_transform=pre_transform)

    if args.perturbed:
        train_loader = apply_perturbation(train_loader, df_train)

    #  Empty dict to store training-testing results
    all_results = {
        "runs": [],
        "aggregated": {}
    }

    base_name = (args.dataset + '_' + args.model + '_epochs_' + str(args.epochs) + '_lr_' + str(args.lr) + '_out_dim_' +
                 str(args.out_dim) + '_hidden_dim_' + str(args.hidden_dim) + '_num_layers_' +
                 str(args.num_layers) + str(args.unknown_classes) + str(args.class_weight) + '_noise_' + str(args.noise))
    if args.pre_transform:
        base_name += '_pretransform_'
    else:
        base_name += '_raw_'

    if args.mahalanobis:
        base_name += '_mahalanobis_'

    best_model_path = './models/weights/' + base_name + '_best_model.pth'

    all_run_dfs = []  # Collect all per-run test outputs

    for run in range(args.num_runs):
        best_f1 = 0.0  # Initialize best_f1, to keep state dict for best model through training
        run_name = f"run_{str(run)}" + base_name  # Initialize run name
        run_best_model_path = './models/weights/' + run_name + '_best_model.pth'

        results_log = {
            "run_name": run_name,
            "train_val_logs": [],
            "test_results": {}
        }

        run_name = base_name  # Convert to the base name, erase previous run numbers from run_name

        print("🧠 Initializing model...")

        if args.model == 'HeteroSage':
            model = HeteroGNN(dim_h=args.hidden_dim, dim_out=args.out_dim, num_layers=args.num_layers).to(device)
        elif args.model == 'MLP':
            model = MLP(input_dim=len(train_loader.dataset[0]['flow'].x[0]), hidden_dim=args.hidden_dim,
                        num_classes=args.out_dim, num_layers=args.num_layers).to(device)
        elif args.model == 'LSTM':
            model = LSTMClassifier(input_dim=len(train_loader.dataset[0]['flow'].x[0]), hidden_dim=args.hidden_dim,
                                   num_layers=args.num_layers, num_classes=args.out_dim).to(device)
        elif args.model == 'Transformer':
            model = TransformerClassifier(input_dim=len(train_loader.dataset[0]['flow'].x[0]),
                                          hidden_dim=args.hidden_dim,
                                          num_heads=8,
                                          num_layers=args.num_layers,
                                          num_classes=args.out_dim).to(device)
        elif args.model == 'HeteroGAT':
            model = HeteroGAT(dim_h=args.hidden_dim, dim_out=args.out_dim, num_layers=args.num_layers, num_heads=4).to(
                device)
        elif args.model == 'HGT':
            model = HGT(dim_h=args.hidden_dim, dim_out=args.out_dim, num_layers=args.num_layers, num_heads=8,
                        metadata=train_loader.dataset[0].metadata()).to(device)
        elif args.model == 'SVM':
            print("Have to implement model!!!!")
            model = SGDOneClassSVM(nu=0.05, shuffle=False)
        else:
            model = None
            print("❌  Model not yet initialized!")

        if args.model != 'SVM':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        weights_tensor = make_class_weight(train_loader, class_weight=args.class_weight, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

        if args.model != 'SVM':
            for epoch in trange(args.epochs + 1, desc="Training Epochs"):
                model.train()
                n_subgraphs = 0
                total_loss = 0

                for batch in train_loader:
                    optimizer.zero_grad()
                    batch.to(device)

                    out = model(batch)
                    loss = criterion(out, batch['flow'].y.argmax(dim=1).long())
                    loss.backward()
                    optimizer.step()
                    n_subgraphs += args.batch_size
                    total_loss += loss.item() * args.batch_size
                if epoch % 10 == 0:
                    val_loss, f1score, temp_pred, temp_true = test(model, device, args.batch_size, val_loader)
                    tqdm.write(
                        f'Epoch {epoch} | Loss: {total_loss / n_subgraphs:.4f} | Val loss: {val_loss:.4f} | Val F1-score: {f1score:.4f}')
                    if f1score > best_f1:
                        best_f1 = f1score
                        torch.save(model.state_dict(), run_best_model_path)
                        tqdm.write(f"Best model saved at epoch {epoch} with F1: {best_f1:.4f}")

                    report_table = cls_report(temp_true, temp_pred, known_labels)
                    results_log["train_val_logs"].append({
                        "epoch": epoch,
                        "val_f1_score": f1score,
                        "val_loss": val_loss,
                        "classification_report": report_table
                    })

            print("🧪 Testing on best model...")
            model.load_state_dict(torch.load(run_best_model_path))
            model.eval()
            print(f"Loaded best model with F1: {best_f1:.4f}")

            _, _, y_pred, y_true = test(model, device, args.batch_size, test_loader, compute_loss=False)
            if args.mahalanobis:
                train_embeds, train_labels = extract_flow_embeddings(model, train_loader, device)
                benign_mask = (train_labels == 0)
                benign_embeds = train_embeds[benign_mask]

                cov = EmpiricalCovariance().fit(benign_embeds.numpy())
                mu = torch.tensor(cov.location_, dtype=torch.float32).to(device)
                sigma_inv = torch.tensor(cov.precision_, dtype=torch.float32).to(device)

                dists_train = mahalanobis_distance(torch.tensor(benign_embeds).to(device), mu, sigma_inv)
                threshold = torch.quantile(dists_train, 0.95).item()

                _, dists_test, f1score, y_pred_maha, y_true_maha, open_acc = test_mahalanobis(
                    model,
                    device,
                    args.batch_size,
                    test_loader,
                    compute_loss=False,
                    mu=mu,
                    sigma_inv=sigma_inv,
                    threshold=threshold,
                    benign_class_index=0  # adjust if needed
                )
                print(f"Final Closed-set F1: {f1score:.4f} | Open-set accuracy: {open_acc:.4f}")
                assert torch.equal(y_true, y_true_maha)  # Make sure we save results in the same order
            # else:
            #     _, _, y_pred, y_true = test(model, device, args.batch_size, test_loader, compute_loss=False)

        else:
            print("🔧 Training One-Class SVM with RBF feature map")

            # --- initialize feature map and model ---
            rbf_feature = RBFSampler(gamma=0.1, n_components=500)

            # --- fit RBF sampler on first batch ---
            first_batch = next(iter(train_loader))['flow'].x.cpu().numpy()
            rbf_feature.fit(first_batch)

            # optional: warm start for model
            model.partial_fit(rbf_feature.transform(first_batch))

            # EPOCH LOOP

            for epoch in range(args.epochs):
                start_time = time.time()
                batches = list(train_loader)
                random.shuffle(batches)
                for batch in batches:

                    # get batch data
                    X = batch['flow'].x.cpu().numpy()
                    y = batch['flow'].y.cpu().numpy()
                    y = np.argmax(y, axis=1)  # one-hot → class index

                    # use *only* benign samples for OCSVM
                    mask = (y == 0)
                    if not np.any(mask):
                        continue

                    X_normal = X[mask]
                    X_rbf = rbf_feature.transform(X_normal)

                    # one incremental update
                    model.partial_fit(X_rbf)

                epoch_time = time.time() - start_time
                print(f"Epoch {epoch + 1}/{args.epochs} completed in {epoch_time:.2f} seconds.")

            # # Fit the transformer on one batch first
            # rbf_feature = RBFSampler(gamma=0.1, n_components=1000)
            #
            # first_batch = next(iter(train_loader))['flow'].x
            # x_rbf = rbf_feature.fit_transform(first_batch)
            # model.partial_fit(x_rbf)
            # for batch in train_loader:
            #     x = batch['flow'].x.cpu().numpy()
            #     y = batch['flow'].y.cpu().numpy()
            #     y = np.argmax(y, axis=1)  # one-hot to class indices
            #
            #     # Only train with benign samples
            #     mask = (y == 0)
            #     if not np.any(mask):
            #         continue  # skip batches without normal data
            #
            #     x_normal = x[mask]
            #
            #     # Transform and train on only normal samples
            #     x_rbf = rbf_feature.transform(x_normal)
            #     model.partial_fit(x_rbf)

            scores_val, auc_val, acc_val, y_pred_val, y_true_val, y_actual_val, report_val = test_ocsvm(model,
                                                                                                        rbf_feature,
                                                                                                        val_loader)
            print(auc_val, acc_val)
            print(report_val)

            scores_test, auc_test, acc_test, y_pred, y_true, y_actual, report = test_ocsvm(model, rbf_feature,
                                                                                           test_loader)
            print(auc_test, acc_test)
            print(report)
            print(auc_test, acc_test)
            y_pred = np.array(y_pred)
            y_pred = np.where(y_pred == 1, 0, 1)

        # _, _, y_pred, y_true = test(model, device, args.batch_size, test_loader, compute_loss=False)
        if args.model == 'SVM':
            df_run = pd.DataFrame({
                'true_label': y_actual,  # SVM branch
                'svm_true_label': y_true,
                'predicted_label': y_pred,
                'score': -scores_test  # AUROC needs high scores for anomaly, low for benign
            })
            all_results = test_report(torch.tensor(y_actual), torch.tensor(y_pred), all_results, results_log, labels,
                                      new_order_labels, args)

        else:
            df_run = pd.DataFrame({
                'true_label': y_true,
                'predicted_label': y_pred
            })
            all_results = test_report(torch.tensor(y_true), torch.tensor(y_pred), all_results, results_log, labels,
                                      new_order_labels, args)
        # Optional: include anomaly or distance metrics if available
        if args.mahalanobis:
            df_run = pd.DataFrame({
                'true_label': y_true,
                'predicted_label': y_pred,
                'y_pred_maha': y_pred_maha,
                'mahalanobis_score': dists_test,
                'threshold': [threshold] * len(y_pred)
            })

        all_run_dfs.append(df_run)
        # save_results(all_results, filename=f"{base_name}_results.json")

        print("✅  Training complete. Results saved.")

    if 'actual_label' in all_run_dfs[0].columns:
        final_df = all_run_dfs[0][['true_label', 'svm_true_label']].copy()
    else:
        final_df = all_run_dfs[0][['true_label']].copy()

    for i, df_run in enumerate(all_run_dfs, start=1):
        final_df[f'predicted_label_{i}'] = df_run['predicted_label']
        if 'score' in df_run.columns:
            final_df[f'score_{i}'] = df_run['score']
        if 'mahalanobis_score' in df_run.columns:
            final_df[f'mahalanobis_score_{i}'] = df_run['mahalanobis_score']
        if 'threshold' in df_run.columns:
            final_df[f'threshold_{i}'] = df_run['threshold']
        if 'y_pred_maha' in df_run.columns:
            final_df[f'y_pred_maha_{i}'] = df_run['y_pred_maha']

    final_df.to_parquet(f'results/dataframes/benchmark_{base_name}.parquet')
    print("✅ Combined results saved:", final_df.shape)
