import os
import json
from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, \
    precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

import numpy as np
import torch

from collections import defaultdict


def per_unknown_results(all_results) -> dict:
    # Containers for metrics
    macro_accs = []
    weighted_accs = []
    overall_unk_accs = []
    per_unknown_class_agg = defaultdict(list)

    # Gather values
    for run_result in all_results["runs"]:
        test_results = run_result["test_results"]
        macro_accs.append(test_results["macro_avg_accuracy"])
        weighted_accs.append(test_results["weighted_avg_accuracy"])
        overall_unk_accs.append(test_results["overall_unknown_accuracy"])

        for cls, acc in test_results["per_unknown_class_accuracy"].items():
            per_unknown_class_agg[cls].append(acc)

    # Compute aggregated stats
    all_results["aggregated"]["macro_avg_accuracy"] = {
        "mean": np.mean(macro_accs),
        "std": np.std(macro_accs)
    }
    all_results["aggregated"]["weighted_avg_accuracy"] = {
        "mean": np.mean(weighted_accs),
        "std": np.std(weighted_accs)
    }
    all_results["aggregated"]["overall_unknown_accuracy"] = {
        "mean": np.mean(overall_unk_accs),
        "std": np.std(overall_unk_accs)
    }

    # Per-class unknown acc
    all_results["aggregated"]["per_unknown_class_accuracy"] = {
        cls: {
            "mean": np.mean(accs),
            "std": np.std(accs)
        }
        for cls, accs in per_unknown_class_agg.items()
    }

    return all_results


def save_results(results_dict, filename=None):
    os.makedirs("results", exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"

    filepath = os.path.join("results", filename)
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"💾  Results saved to {filepath}")
    return filepath


def cls_report(true, pred, target_names):
    class_report = classification_report(true,
                                         pred,
                                         target_names=target_names,
                                         zero_division=0,
                                         output_dict=True)
    report_table = []

    for label, metrics in class_report.items():
        # skip if label is 'accuracy' (not a dict)
        if isinstance(metrics, dict):
            report_table.append([
                label,  # class name or 'macro avg', etc.
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1-score", 0),
                metrics.get("support", 0),
            ])

    return report_table


def test_report(y_true, y_pred, all_results, results_log, labels, new_order_labels, args):
    if args.unknown_classes is not None:
        # Map labels to indices
        label_to_idx = {label: i for i, label in enumerate(new_order_labels)}
        unknown_idxs = [label_to_idx[l] for l in args.unknown_classes]
        benign_idx = label_to_idx['benign']

        # Select unknown samples
        # mask = np.isin(y_true.cpu(), unknown_idxs)

        # correct = ((y_pred[mask] != benign_idx).sum().item()) if mask.any() else 0
        # total = mask.sum().item()
        # unknown_acc = correct / total if total else 0.0
        # print(correct, total, unknown_acc)

        unknown_class_accs = {}
        correct_total = 0
        total_total = 0

        for unknown_label in args.unknown_classes:
            idx = label_to_idx[unknown_label]
            class_mask = (y_true.cpu().numpy() == idx)
            class_total = class_mask.sum()
            if class_total == 0:
                class_acc = 0.0
            else:
                class_correct = (y_pred.cpu().numpy()[class_mask] != benign_idx).sum()
                class_acc = class_correct / class_total

            unknown_class_accs[unknown_label] = class_acc
            correct_total += class_correct
            total_total += class_total

        # macro and weighted averages
        macro_acc = np.mean(list(unknown_class_accs.values()))
        weighted_acc = np.average(
            list(unknown_class_accs.values()),
            weights=[(y_true.cpu().numpy() == label_to_idx[label]).sum() for label in args.unknown_classes]
        )

        # original all-inclusive unknown_acc (you computed earlier)
        # note: if you prefer the original version: correct = (y_pred[mask] != benign_idx) etc.
        unknown_acc = correct_total / total_total if total_total else 0.0

        print(f"Per-unknown-class accuracy: {unknown_class_accs}")
        print(f"Macro avg accuracy: {macro_acc:.4f}")
        print(f"Weighted avg accuracy: {weighted_acc:.4f}")
        print(f"All-inclusive unknown acc: {unknown_acc:.4f}")

        final_report = classification_report(y_true,
                                             y_pred,
                                             labels=list(range(len(new_order_labels))),
                                             target_names=new_order_labels,
                                             digits=4,
                                             zero_division=0,
                                             output_dict=True)
        print(final_report)

        # monitor predictions for each class
        final_table = []
        for label, metrics in final_report.items():
            # skip if label is 'accuracy' (not a dict)
            if isinstance(metrics, dict):
                final_table.append([
                    label,  # class name or 'macro avg', etc.
                    metrics.get("precision", 0),
                    metrics.get("recall", 0),
                    metrics.get("f1-score", 0),
                    metrics.get("support", 0),
                ])

        conf_matrix = confusion_matrix(y_true.cpu(), y_pred.cpu()).tolist()

        results_log["test_results"] = {
            "classification_report": final_table,
            "confusion_matrix": conf_matrix,
            "per_unknown_class_accuracy": unknown_class_accs,
            "macro_avg_accuracy": macro_acc,
            "weighted_avg_accuracy": weighted_acc,
            "overall_unknown_accuracy": unknown_acc
        }

        # run.finish()
        all_results["runs"].append(results_log)
        all_results = per_unknown_results(all_results)
        print("============ Aggregated Results ============")
        print(json.dumps(all_results["aggregated"], indent=4))

    else:
        print("❗  No open-set was defined. Reporting close-set classification results")
        final_report = classification_report(y_true,
                                             y_pred,
                                             target_names=labels,
                                             digits=4,
                                             zero_division=0,
                                             output_dict=True)
        # monitor predictions for each class
        final_table = []
        for label, metrics in final_report.items():
            # skip if label is 'accuracy' (not a dict)
            if isinstance(metrics, dict):
                final_table.append([
                    label,  # class name or 'macro avg', etc.
                    metrics.get("precision", 0),
                    metrics.get("recall", 0),
                    metrics.get("f1-score", 0),
                    metrics.get("support", 0),
                ])

        conf_matrix = confusion_matrix(y_true.cpu(), y_pred.cpu()).tolist()

        results_log["test_results"] = {"classification_report": final_table,
                                       "confusion_matrix": conf_matrix}

        # run.finish()
        all_results["runs"].append(results_log)
        print("============ Test Results on Closed Set ============")
        print(results_log["test_results"])
    return all_results


def plot_auroc(y_true_anomaly, scores_anomaly, auroc):
    # --- Plotting the ROC Curve ---
    # Calculate FPR and TPR for various thresholds
    fpr, tpr, thresholds = roc_curve(y_true_anomaly, scores_anomaly)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')  # No-skill classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('figures/auroc_bruteforce.png', dpi=300)
    plt.close()


def calc_auroc(analysis_data, target_tpr=0.99):
    # Map val < 0 to 0 and val > 1 to 1
    analysis_data['anomaly_score'] = analysis_data['anomaly_score'].clip(0, 1)

    y_true = analysis_data['true_label']
    scores_anomaly = analysis_data['anomaly_score']

    if torch.is_tensor(y_true):
        y_true_anomaly = (y_true != 0).long().cpu().numpy()
    else:
        y_true_anomaly = (y_true != 0).astype(int)

    auroc = roc_auc_score(y_true_anomaly, scores_anomaly)
    auprc = average_precision_score(y_true_anomaly, scores_anomaly)
    print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

    plot_auroc(y_true_anomaly, scores_anomaly, auroc)

    # Find threshold for 99% TPR
    prec, rec, thresh = precision_recall_curve(y_true_anomaly, scores_anomaly)

    idx = (rec >= target_tpr).nonzero()[0][0]  # first index where TPR >= 95%
    threshold = thresh[idx]
    print(f"Threshold for {target_tpr} TPR: {threshold:.4f}")

    if torch.is_tensor(scores_anomaly):
        y_pred_anomaly = (scores_anomaly >= threshold).long().cpu().numpy()
    else:
        y_pred_anomaly = (scores_anomaly >= threshold).astype(int)

    results = classification_report(y_true_anomaly, y_pred_anomaly, target_names=["benign", "attack"])
    print(results)

    return auroc, auprc, threshold
