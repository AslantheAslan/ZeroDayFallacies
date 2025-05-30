import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.preprocess import preprocess_CIDDS
from data.loader import create_dataloader
from data.get_dataset import get_dataset

from models.hetero_gnn import HeteroGNN
from models.mlp import MLP
from models.lstm import LSTMClassifier
from models.transformer import TransformerClassifier


from training.test import test

from attacks.random_attack import apply_perturbation

from utils.results import save_results, cls_report, test_report

from data.download import download_CIDDS

import os
import pandas as pd
from tqdm import trange

from sklearn.metrics import f1_score, classification_report, confusion_matrix


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

    print("📤 Creating data loaders...")
    train_loader = create_dataloader(df_train, known_labels, args.batch_size, features_flow, features_host)
    val_loader = create_dataloader(df_val, known_labels, args.batch_size, features_flow, features_host)
    test_loader = create_dataloader(df_test, new_order_labels, args.batch_size, features_flow, features_host)

    if args.perturbed:
        train_loader = apply_perturbation(train_loader, df_train)

    #  Empty dict to store training-testing results
    all_results = {
        "runs": [],
        "aggregated": {}
    }

    base_name = (args.model + '_epochs_' + str(args.epochs) + '_lr_' + str(args.lr) + '_out_dim_' +
                 str(args.out_dim) + '_hidden_dim_' + str(args.hidden_dim) + '_num_layers_' +
                 str(args.num_layers) + str(args.unknown_classes))
    best_model_path = './models/weights/' + base_name + '_best_model.pth'

    best_f1 = 0.0  # Initialize best_f1, to keep state dict for best model through training
    for run in range(args.num_runs):
        run_name = f"run_{str(run)}" + base_name  # Initialize run name

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
            model = MLP(input_dim=len(train_loader.dataset[0]['flow'].x[0]), hidden_dim=args.hidden_dim, num_classes=args.out_dim).to(device)
        elif args.model == 'LSTM':
            model = LSTMClassifier(input_dim=len(train_loader.dataset[0]['flow'].x[0]), hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_classes=args.out_dim).to(device)
        elif args.model == 'Transformer':
            model = TransformerClassifier(input_dim=len(train_loader.dataset[0]['flow'].x[0]),
                                          hidden_dim=args.hidden_dim,
                                          num_heads=8,
                                          num_layers=args.num_layers,
                                          num_classes=args.out_dim).to(device)
        else:
            model = None
            print("❌  Model not yet initialized!")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Burada kaldim
        best_val_acc = 0
        best_model_state = None

        beta_values = []

        for epoch in trange(args.epochs + 1, desc="Training Epochs"):
            model.train()
            n_subgraphs = 0
            total_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                batch.to(device)

                out = model(batch)
                loss = F.cross_entropy(out, batch['flow'].y.argmax(dim=1).long())
                loss.backward()
                optimizer.step()
                n_subgraphs += args.batch_size
                total_loss += float(loss) * args.batch_size
            if 'beta' in model.parameters():
                beta_values.append(model.beta.item())
                tqdm.write(f"Beta value: {model.beta.item()}")
            if epoch % 10 == 0:
                val_loss, f1score, temp_pred, temp_true = test(model, device, args.batch_size, val_loader)
                tqdm.write(
                    f'Epoch {epoch} | Loss: {total_loss / n_subgraphs:.4f} | Val loss: {val_loss:.4f} | Val F1-score: {f1score:.4f}')
                if f1score > best_f1:
                    best_f1 = f1score
                    torch.save(model.state_dict(), best_model_path)
                    tqdm.write(f"Best model saved at epoch {epoch} with F1: {best_f1:.4f}")

                report_table = cls_report(temp_true, temp_pred, known_labels)
                results_log["train_val_logs"].append({
                    "epoch": epoch,
                    "val_f1_score": f1score,
                    "val_loss": val_loss,
                    "classification_report": report_table
                })

        print("🧪 Testing on best model...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        print(f"Loaded best model with F1: {best_f1:.4f}")

        _, _, y_pred, y_true = test(model, device, args.batch_size, test_loader, compute_loss=False)
        all_results = test_report(y_true, y_pred, all_results, results_log, labels, new_order_labels, args)

        save_results(all_results, filename=f"{base_name}_results.json")

        print("✅  Training complete. Results saved.")
