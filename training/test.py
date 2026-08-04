import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
import numpy as np

from training.mahalanobis import mahalanobis_distance
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def test(model, device, batch_size, loader, compute_loss=True, verbose=False):
    model.eval()
    y_pred = []
    y_true = []
    n_subgraphs = 0
    total_loss = 0

    if verbose:
        entropies = []
        anomaly_scores_list = []
        gate_scores_list = []

    for batch in loader:
        batch.to(device)
        if not verbose:
            out = model(batch)
        else:
            out, anomaly, gate = model.forward_with_anomaly(batch)
            probs = F.softmax(out, dim=1)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)  # Avoid log(0) -> -inf case
            #  map -0.0 (near-zeros) to -1.0 bcs -0.0 usually observed for benign
            entropy = torch.where(entropy.abs() < 1e-12, torch.tensor(-1.0, device=entropy.device), entropy)
        if compute_loss:
            # loss = F.cross_entropy(out, batch['flow'].y.float())
            loss = F.cross_entropy(out, batch['flow'].y.argmax(dim=1).long())
            total_loss += float(loss) * batch_size

        y_pred.append(out.argmax(dim=1))
        y_true.append(batch['flow'].y.argmax(dim=1))
        n_subgraphs += batch_size

        if verbose:
            entropies.append(entropy)
            print(f'Length of entropies list: {len(entropies)}')
            anomaly_scores_list.append(anomaly)
            gate_scores_list.append(gate)
            print(f'Entropies shape: {entropies[0].shape}\nAnomaly shape: {anomaly_scores_list[0].shape}')
    y_pred = torch.cat(y_pred).cpu()
    y_true = torch.cat(y_true).cpu()
    f1score = f1_score(y_true, y_pred, average='macro')

    if verbose:
        entropies = torch.cat(entropies).cpu()
        anomaly_scores_list = torch.cat(anomaly_scores_list).squeeze(1).cpu()
        gate_scores_list = torch.cat(gate_scores_list).squeeze(1).cpu()

        return total_loss / n_subgraphs, f1score, y_pred, y_true, anomaly_scores_list, gate_scores_list, entropies

    return total_loss / n_subgraphs, f1score, y_pred, y_true

@torch.no_grad()
def test_mahalanobis(model, device, batch_size, loader, compute_loss=True,
                     mu=None, sigma_inv=None, threshold=None, benign_class_index=0):
    """
    If mu, sigma_inv, and threshold are provided, the function performs Mahalanobis-based
    open-set detection in addition to normal classification.

    Args:
        mu: Mean embedding vector of the benign class (Tensor of shape [D])
        sigma_inv: Inverse covariance matrix of benign embeddings (Tensor of shape [D, D])
        threshold: Distance threshold above which a sample is considered NOT benign
        benign_class_index: Index of the benign class in one-hot encoding
    """

    model.eval()
    y_pred = []
    y_true = []
    open_set_flags = []
    n_subgraphs = 0
    total_loss = 0
    dists_all = []
    for batch in loader:
        batch.to(device)
        out = model(batch)  # shape: [N, num_classes]

        if compute_loss:
            loss = F.cross_entropy(out, batch['flow'].y.argmax(dim=1).long())
            total_loss += float(loss) * batch_size

        preds = out.argmax(dim=1)  # closed-set predictions
        labels = batch['flow'].y.argmax(dim=1)

        y_pred.append(preds)
        y_true.append(labels)

        # If Mahalanobis open-set detection is enabled:
        if mu is not None and sigma_inv is not None and threshold is not None:
            if hasattr(model, 'get_flow_embeddings'):
                flow_emb = model.get_flow_embeddings(batch)
            else:
                # fallback: use output before final classifier if applicable
                raise ValueError("Model must expose get_flow_embeddings(batch) to extract embeddings.")

            # Compute Mahalanobis distances
            dists = mahalanobis_distance(flow_emb, mu, sigma_inv)
            dists_all.append(dists)
            is_outlier = (dists > threshold).long()

            # Binary open-set prediction:
            # 1 = correctly flagged as attack (not benign), 0 = mistakenly flagged as benign
            predicted_benign = (preds == benign_class_index).long()
            false_benign = (is_outlier == 0) & (labels != benign_class_index)
            open_set_flags.append(1 - false_benign.long())  # 1 = correct rejection, 0 = incorrect

        n_subgraphs += batch_size

    y_pred = torch.cat(y_pred).cpu()
    y_true = torch.cat(y_true).cpu()
    dists_all = torch.cat(dists_all).cpu()
    f1score = f1_score(y_true, y_pred, average='macro')

    if open_set_flags:
        open_set_accuracy = torch.cat(open_set_flags).float().mean().item()
        return total_loss / n_subgraphs, dists_all, f1score, y_pred, y_true, open_set_accuracy

    return total_loss / n_subgraphs, dists_all, f1score, y_pred, y_true


def test_ocsvm(model, rbf_feature, loader, inlier_class=0):
    """
    Evaluate SGDOneClassSVM + RBFSampler on one-hot encoded labels.

    Args:
        model: Trained SGDOneClassSVM
        rbf_feature: Fitted RBFSampler
        loader: DataLoader with 'flow'.x and one-hot 'flow'.y
        inlier_class: The class to treat as inlier (default: 0)

    Returns:
        scores: Decision function values
        auc: ROC AUC
        acc: Accuracy
        y_pred: Predicted labels (+1 or -1)
        y_true: Ground truth labels (+1 or -1)
    """
    y_pred = []
    y_true, y_actual = [], []
    all_scores = []

    for batch in loader:
        X = batch['flow'].x.cpu().numpy()
        y = batch['flow'].y.cpu().numpy()
        y = np.argmax(y, axis=1)  # Convert one-hot to class indices
        y_actual.extend(np.array(y))

        y = np.where(y == inlier_class, 1, -1)  # Map to +1 / -1

        X_rbf = rbf_feature.transform(X)
        scores = model.decision_function(X_rbf)  # Used for AUC
        preds = model.predict(X_rbf)  # +1 or -1

        y_pred.append(preds)
        y_true.append(y)
        all_scores.append(scores)

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    all_scores = np.concatenate(all_scores)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, all_scores)

    report = classification_report(y_true, y_pred,
                                   target_names=['Anomaly', 'Normal'])

    return all_scores, auc, acc, y_pred, y_true, y_actual, report
