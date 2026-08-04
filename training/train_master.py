import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import Compose, RemoveIsolatedNodes, AddSelfLoops, NormalizeFeatures
from tqdm import tqdm

from data.preprocess import preprocess_CIDDS
from data.loader import create_dataloader, GaussianNoise, build_bipartite_adjacency, create_full_graph
from data.get_dataset import get_dataset

from models.hetero_gnn import HeteroGNN, HeteroGAT, HGT
from models.mlp import MLP, meta_MLP, GatedFusionMetaMLP, ExtendedGatedFusionMetaMLP
from models.lstm import LSTMClassifier
from models.transformer import TransformerClassifier

from utils.results import save_results, cls_report, test_report, calc_auroc
from training.test import test

import os
import pandas as pd
from tqdm import trange
from utils.utils import make_class_weight, get_model_size, visualize_analysis

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


def select_embedding_model(train_loader, args, device):
    print(f'Embedding model: {args.model}')
    if args.model == 'HeteroSage':
        model_arch = HeteroGNN(dim_h=args.hidden_dim, dim_out=args.out_dim, num_layers=args.num_layers).to(device)
    elif args.model == 'MLP':
        model_arch = MLP(input_dim=len(train_loader.dataset[0]['flow'].x[0]), hidden_dim=args.hidden_dim,
                         num_classes=args.out_dim, num_layers=args.num_layers).to(device)
    elif args.model == 'LSTM':
        model_arch = LSTMClassifier(input_dim=len(train_loader.dataset[0]['flow'].x[0]), hidden_dim=args.hidden_dim,
                                    num_layers=args.num_layers, num_classes=args.out_dim).to(device)
    elif args.model == 'HeteroGAT':
        model_arch = HeteroGAT(dim_h=args.hidden_dim, dim_out=args.out_dim, num_layers=args.num_layers, num_heads=4).to(
            device)
    elif args.model == 'HGT':
        model_arch = HGT(dim_h=args.hidden_dim, dim_out=args.out_dim, num_layers=args.num_layers, num_heads=8,
                         metadata=train_loader.dataset[0].metadata()).to(device)
    else:
        return NotImplementedError("The model type is not supported")

    return model_arch


def load_model(train_loader, model_path, args, device):
    model = select_embedding_model(train_loader, args, device)
    print(f'Model loaded from: {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model


class GraphDataset(Dataset):
    def __init__(self, graphs): self.graphs = graphs

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx): return self.graphs[idx]


def pretrained_embedding(train_loader, test_loader, test_loader_raw, model_path, raw_model_path, args, device):
    pretrained_model = load_model(train_loader, model_path, args, device)
    raw_pretrained_model = load_model(train_loader, raw_model_path, args, device)

    embedded_batches = []
    for batch, raw_batch in zip(test_loader, test_loader_raw):  # or raw_loader if separate
        batch, raw_batch = batch.to(device), raw_batch.to(device)

        emb = pretrained_model.get_flow_embeddings(batch)
        raw_emb = raw_pretrained_model.get_flow_embeddings(raw_batch)

        # Store them as new node attributes (node-level features)
        batch['flow'].x_emb = emb.detach()
        batch['flow'].x_emb_raw = raw_emb.detach()

        batch['flow'].x_delta = (emb - raw_emb).detach()

        cos_sim = F.cosine_similarity(emb, raw_emb, dim=-1, eps=1e-8)  # shape: (num_nodes,)
        batch['flow'].x_cos = cos_sim.unsqueeze(-1).detach()  # make it (num_nodes, 1)

        embedded_batches.append(batch.cpu())  # keep on CPU to save GPU RAM
    return embedded_batches


def run_master_training(args):
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
    # args.noise = 0.05
    pre_transform = Compose([
        RemoveIsolatedNodes(),  # remove unconnected nodes
        AddSelfLoops(),  # ensure self-messages
        NormalizeFeatures(),  # normalize each node type's features
        GaussianNoise(std=args.noise)  # add small noise to features
    ])

    print("📤 Creating data loaders...")
    train_loader = create_dataloader(df_train, known_labels, args.batch_size, features_flow, features_host,
                                     pre_transform=pre_transform)
    val_loader = create_dataloader(df_val, known_labels, args.batch_size, features_flow, features_host,
                                   pre_transform=pre_transform)
    test_loader = create_dataloader(df_test, new_order_labels, args.batch_size, features_flow, features_host,
                                    pre_transform=pre_transform)

    raw_train_loader = create_dataloader(df_train, known_labels, args.batch_size, features_flow, features_host)
    raw_val_loader = create_dataloader(df_val, known_labels, args.batch_size, features_flow, features_host)
    raw_test_loader = create_dataloader(df_test, new_order_labels, args.batch_size, features_flow,
                                        features_host)

    # Input embeddings for the meta-model to learn
    train_emb = pretrained_embedding(train_loader, train_loader, raw_train_loader, args.path, args.path_raw, args,
                                     device)  # This is the correct utilization

    # Input embeddings for the meta-model to validate and select the best version
    val_emb = pretrained_embedding(val_loader, val_loader, raw_val_loader, args.path, args.path_raw, args, device)

    train_loader_emb = DataLoader(GraphDataset(train_emb), batch_size=args.batch_size, shuffle=False)
    val_loader_emb = DataLoader(GraphDataset(val_emb), batch_size=args.batch_size)

    #  Empty dict to store training-testing results
    all_results = {
        "runs": [],
        "aggregated": {}
    }

    base_name = (args.dataset + args.model + '_epochs_' + str(args.epochs) + '_lr_' + str(args.lr) + '_out_dim_' +
                 str(args.out_dim) + '_hidden_dim_' + str(args.hidden_dim) + '_num_layers_' +
                 str(args.num_layers) + str(args.unknown_classes) + str(args.class_weight) + '_noise_' + str(args.noise) + f'{args.master_arch}' +
                 '_meta_')
    best_model_path = './models/weights/' + base_name + '_best_model.pth'

    all_run_dfs = []
    for run in range(args.num_runs):
        best_f1 = 0.0  # Initialize best_f1, to keep state dict for best model through training
        run_name = f"run_{str(run)}" + base_name  # Initialize run name

        results_log = {
            "run_name": run_name,
            "train_val_logs": [],
            "test_results": {}
        }

        print("Initializing the meta model, a simple MLP, on embeddings")
        # print(f"Dataset obj length {len(train_emb.dataset)}")
        # print(f"Vector length for embeddings: {len(train_emb[0]['flow'].emb[0])}+{len(train_emb[0]['flow'].raw_emb[0])}")


        if args.master_arch == 'MLP':
            meta_model = meta_MLP(input_dim=193,
                                  hidden_dim=args.hidden_dim,
                                  num_classes=args.out_dim,
                                  num_layers=3).to(device)
        elif args.master_arch == 'GatedMLP':
            meta_model = GatedFusionMetaMLP(emb_raw_dim=64,
                                          emb_dim=64,
                                          hidden_dim=args.hidden_dim,
                                          num_classes=args.out_dim,
                                          num_layers=3,
                                          fusion_type="simple").to(device)
        elif args.master_arch == 'ExtendedGatedMLP':
            meta_model = ExtendedGatedFusionMetaMLP(emb_raw_dim=64,
                                                    emb_dim=64,
                                                    cos_dim=1,
                                                    delta_dim=64,
                                                    hidden_dim=args.hidden_dim,
                                                    num_classes=args.out_dim,
                                                    num_layers=3).to(device)
        else:
            print("The model type is not supported")


        optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.lr)

        weights_tensor = make_class_weight(train_loader, class_weight=args.class_weight, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

        beta_values = []
        for epoch in trange(args.epochs + 1, desc="Training Epochs"):
            meta_model.train()
            n_subgraphs = 0
            total_loss = 0

            for batch in train_loader_emb:
                optimizer.zero_grad()
                batch.to(device)

                # out = meta_model(batch)
                out, anomaly_score, gate_values = meta_model.forward_with_anomaly(batch)
                cls_loss = criterion(out, batch['flow'].y.argmax(dim=1).long())

                #  loss = criterion(out, batch['flow'].y.argmax(dim=1).long())  # The original was this line
                #  loss.backward()  # The original loss.backward() was here. Later I added the joint loss
                #  Jointly optimize anomaly head and classification head

                true_labels = batch['flow'].y.argmax(dim=1).long()
                anomaly_labels = (true_labels != 0).float()

                # Anomaly loss
                anomaly_loss = F.binary_cross_entropy(anomaly_score.squeeze(), anomaly_labels)

                # Gate regularization, since gate vals approximate to very different values in every train run
                entropy = - (gate_values * torch.log(gate_values + 1e-8)
                             + (1 - gate_values) * torch.log(1 - gate_values + 1e-8))
                gate_entropy = entropy.mean()

                # Combine
                loss = cls_loss + 0.1 * anomaly_loss + 0.1 * gate_entropy

                loss.backward()
                optimizer.step()
                n_subgraphs += args.batch_size
                total_loss += float(loss) * args.batch_size

            if 'beta' in meta_model.parameters():
                beta_values.append(meta_model.beta.item())
                tqdm.write(f"Beta value: {meta_model.beta.item()}")
            if epoch % 10 == 0:
                val_loss, f1score, temp_pred, temp_true = test(meta_model, device, args.batch_size, val_loader_emb, compute_loss=True, verbose=False)
                tqdm.write(
                    f'Epoch {epoch} | Loss: {total_loss / n_subgraphs:.4f} | Val loss: {val_loss:.4f} | Val F1-score: {f1score:.4f}')
                if f1score > best_f1:
                    best_f1 = f1score
                    torch.save(meta_model.state_dict(), best_model_path)
                    tqdm.write(f"Best model saved at epoch {epoch} with F1: {best_f1:.4f}")

                report_table = cls_report(temp_true, temp_pred, known_labels)
                results_log["train_val_logs"].append({
                    "epoch": epoch,
                    "val_f1_score": f1score,
                    "val_loss": val_loss,
                    "classification_report": report_table
                })

        print("🧪 Testing on best model...")
        meta_model.load_state_dict(torch.load(best_model_path))
        meta_model.eval()
        print(f"Loaded best model with F1: {best_f1:.4f}")

        test_emb = pretrained_embedding(test_loader, test_loader, raw_test_loader, args.path, args.path_raw, args,
                                        device)
        test_loader_emb = DataLoader(GraphDataset(test_emb), batch_size=args.batch_size)

        # _, _, y_pred, y_true = test(meta_model, device, args.batch_size, test_loader_emb, compute_loss=False)
        _, _, y_pred, y_true, scores_anomaly, scores_gate, entropies = test(meta_model, device, args.batch_size, test_loader_emb, compute_loss=False, verbose=True)

        analysis_data = {
            'true_label': y_true,
            'predicted_label': y_pred,
            'anomaly_score': scores_anomaly,
            'gate_score': scores_gate,
            'entropy': entropies
        }
        print(type(y_true), type(y_pred), type(scores_anomaly), type(scores_gate))
        print(y_true.shape, y_pred.shape, scores_anomaly.shape, scores_gate.shape)

        df_analysis = pd.DataFrame(analysis_data)

        all_run_dfs.append(df_analysis)

        auroc, auprc, threshold = calc_auroc(analysis_data, target_tpr=0.9999)
        print(f'AUROC: {auroc}')

        # all_results = test_report(torch.tensor(y_true), torch.tensor(y_pred), all_results, results_log, labels,
        #                           new_order_labels, args)

        # save_results(all_results, filename=f"{base_name}_results.json")

        # print("✅  Training complete. Results saved.")

        # visualize_analysis(df_analysis, args)

    final_df = all_run_dfs[0][['true_label']].copy()

    for i, df_run in enumerate(all_run_dfs, start=1):
        final_df[f'predicted_label_{i}'] = df_run['predicted_label']
        final_df[f'anomaly_score_{i}'] = df_run['anomaly_score']
        final_df[f'gate_score_{i}'] = df_run['gate_score']
        final_df[f'entropy_{i}'] = df_run['entropy']

    final_df.to_parquet(f'results/dataframes/{base_name}.parquet')
    print("--- First 5 recorded observations ---")
    print(final_df.head())
