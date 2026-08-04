import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns


def make_class_weight(train_loader, class_weight, device):
    # Collect all training labels
    all_labels = []
    for batch in train_loader:
        y_batch = batch['flow'].y.argmax(dim=1).cpu().numpy()  # assuming one-hot or multi-hot encoding
        all_labels.extend(y_batch)

    all_labels = np.array(all_labels)

    # Compute class weights
    classes = np.unique(all_labels)

    # Check format & convert to dict
    if isinstance(class_weight, list):
        if len(class_weight) != len(classes):
            raise ValueError(
                f"Length of class_weight list ({len(class_weight)}) does not match "
                f"number of classes in data ({len(classes)})."
            )
        class_weight = get_class_weight_dict(weights=class_weight)

    # elif class_weight is None:
    #     class_weight = np.ones(len(classes))

    weights = compute_class_weight(class_weight=class_weight, classes=classes, y=all_labels)

    # Convert to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    return weights_tensor


def parse_class_weight(value):
    print(value)
    if value == 'balanced':
        return 'balanced'
    try:
        # Convert string of floats separated by commas into list
        weights = list(map(float, value.split(',')))
        return weights
    except ValueError:
        raise argparse.ArgumentTypeError("class_weight must be 'balanced' or comma-separated floats (e.g. 1.0,0.5,2.0)")


def get_class_weight_dict(weights):
    """
    Convert a list of weights to {class_id: weight} dict.

    Args:
        weights (list): List of weights, e.g., [1.0, 0.5, 2.0]

    Returns:
        dict: {0: 1.0, 1: 0.5, 2: 2.0}
    """
    if not isinstance(weights, list):
        raise ValueError("weights must be a list of floats")

    return {i: w for i, w in enumerate(weights)}


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # Calculate buffer size
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # Total model size in bytes
    total_size_bytes = param_size + buffer_size

    # Convert to MB
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Model size: {total_size_mb:.2f} MB")


def visualize_analysis(df_results, args):
    # Map val < 0 to 0 and val > 1 to 1
    df_results['anomaly_score'] = df_results['anomaly_score'].clip(0, 1)

    print("\n--- Statistical summary of anomaly scores per label ---")
    print(df_results.groupby('true_label')['anomaly_score'].describe())

    # Histogram
    if df_results['anomaly_score'].nunique() > 1:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df_results,
            x='anomaly_score',
            hue='true_label',
            kde=True,
            bins=50,
            palette='viridis'
        )
        plt.title('Distribution of Anomaly Scores by Ground Truth Label')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.savefig(f'figures/{args.unknown_classes}_anomaly_scores_vs_true_label.png', dpi=300)
        plt.close()
    else:
        print("⚠️ Histogram skipped: not enough variance in anomaly scores.")

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x='true_label', y='anomaly_score', palette='muted')
    plt.title('Anomaly Score Distribution per Ground Truth Label')
    plt.xlabel('Ground Truth Label')
    plt.ylabel('Anomaly Score')
    plt.savefig(f'figures/{args.unknown_classes}_true_label_vs_anomaly_scores.png', dpi=300)
    plt.close()


def plot_3d(df_analysis):

    # 1. Prepare the data
    x = df_analysis['anomaly_score']
    y = df_analysis['gate_score']
    z = df_analysis['entropy']
    labels = df_analysis['true_label']

    # Normalize labels for colormap
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(labels.min(), labels.max())

    # 2. Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Plot all non-class-4 points (transparent)
    mask4 = labels == 4
    ax.scatter(x[~mask4], y[~mask4], z[~mask4],
               c=[cmap(norm(lbl)) for lbl in labels[~mask4]],
               alpha=0.1, s=20)

    # 4. Plot class-4 points again with outline
    ax.scatter(x[mask4], y[mask4], z[mask4],
               c=[cmap(norm(lbl)) for lbl in labels[mask4]],
               alpha=1.0, s=60,
               edgecolor='k', linewidth=0.8,
               label="Class 4 (Unseen Attacks)")

    # 5. Labels and title
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Gate Score')
    ax.set_zlabel('Entropy')
    ax.set_title('3D Visualization of Model Uncertainty Scores')

    # 6. Legend
    import matplotlib.lines as mlines
    legend_handles = []
    for lbl in sorted(labels.unique()):
        color = cmap(norm(lbl))
        if lbl == 4:
            legend_handles.append(
                mlines.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markeredgecolor='k',
                              markersize=10, label="Unseen Attacks")
            )
        else:
            legend_handles.append(
                mlines.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, alpha=0.5,
                              markersize=8, label=f"Class {lbl}")
            )
    ax.legend(handles=legend_handles, title="True Labels", loc="upper right")

    plt.savefig(f'figures/3d_bruteforce_lambda_01.png', dpi=300)
    plt.close()

# import pandas as pd
#
# df = pd.read_parquet('figures/bruteforce_analysis_01.parquet')
# plot_3d(df)
