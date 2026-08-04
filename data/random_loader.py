import numpy as np
import pandas as pd
from collections import Counter
from loader import create_dataloader


def create_balanced_random_labels_for_dataframe(df, label_columns, balance_strategy='uniform'):
    """
    Create balanced random labels for pre-training while preserving dataframe structure.

    Args:
        df: input dataframe
        label_columns: list of label column names (one-hot encoded)
        balance_strategy: 'uniform' for equal distribution

    Returns:
        df_random: dataframe with balanced random labels
        label_mapping: dict showing original to random label mapping stats
    """
    df_random = df.copy()

    # Get the original labels (assuming one-hot encoded)
    original_labels = df[label_columns].values
    original_class_indices = np.argmax(original_labels, axis=1)

    # Get unique classes and their counts
    unique_classes = np.unique(original_class_indices)
    n_classes = len(unique_classes)
    n_samples = len(original_labels)

    # Print original distribution
    original_counts = Counter(original_class_indices)
    print(f"Original class distribution: {dict(original_counts)}")

    # Generate balanced random labels
    if balance_strategy == 'uniform':
        # Equal probability for each class
        random_class_indices = np.random.choice(unique_classes, size=n_samples)
    elif balance_strategy == 'proportional':
        # Maintain some proportion but more balanced
        original_probs = np.array([original_counts[i] for i in unique_classes]) / n_samples
        # Smooth the probabilities to be more balanced
        smoothed_probs = (original_probs + 1 / n_classes) / 2
        smoothed_probs = smoothed_probs / smoothed_probs.sum()
        random_class_indices = np.random.choice(unique_classes, size=n_samples, p=smoothed_probs)

    # Print new distribution
    new_counts = Counter(random_class_indices)
    print(f"Random balanced class distribution: {dict(new_counts)}")

    # Convert back to one-hot encoding
    random_labels = np.zeros((n_samples, len(label_columns)))
    for i, class_idx in enumerate(random_class_indices):
        if class_idx < len(label_columns):
            random_labels[i, class_idx] = 1

    # Replace original labels with random balanced labels
    for i, col in enumerate(label_columns):
        if i < random_labels.shape[1]:
            df_random[col] = random_labels[:, i]

    # Create mapping info
    label_mapping = {
        'original_distribution': dict(original_counts),
        'random_distribution': dict(new_counts),
        'balance_improvement': {
            'original_max_ratio': max(original_counts.values()) / min(original_counts.values()),
            'random_max_ratio': max(new_counts.values()) / min(new_counts.values())
        }
    }

    return df_random, label_mapping


def create_dataloader_with_mode(df, labels, BATCH_SIZE, features_flow, features_host,
                                subgraph_size=1024, mode='original', balance_strategy='uniform'):
    """
    Enhanced dataloader that can create loaders for both original and random labels.

    Args:
        df: input dataframe
        labels: list of label column names
        BATCH_SIZE: batch size
        features_flow: flow feature column names
        features_host: host feature column names
        subgraph_size: size of each subgraph
        mode: 'original' for original labels, 'random' for balanced random labels
        balance_strategy: strategy for random label generation

    Returns:
        DataLoader object
        label_info: dict with label information (only for random mode)
    """
    if mode == 'random':
        print(f"Creating dataloader with balanced random labels using {balance_strategy} strategy...")
        df_processed, label_info = create_balanced_random_labels_for_dataframe(df, labels, balance_strategy)
        dataloader = create_dataloader(df_processed, labels, BATCH_SIZE, features_flow, features_host, subgraph_size)
        return dataloader, label_info
    else:
        print("Creating dataloader with original labels...")
        dataloader = create_dataloader(df, labels, BATCH_SIZE, features_flow, features_host, subgraph_size)
        return dataloader, None


def analyze_class_distribution(df, label_columns):
    """
    Analyze and print class distribution statistics.

    Args:
        df: input dataframe
        label_columns: list of label column names
    """
    # Get class indices from one-hot encoded labels
    labels = df[label_columns].values
    class_indices = np.argmax(labels, axis=1)

    # Count occurrences
    unique_classes, counts = np.unique(class_indices, return_counts=True)

    print("\nClass Distribution Analysis:")
    print("-" * 50)
    total_samples = len(df)
    for i, (class_idx, count) in enumerate(zip(unique_classes, counts)):
        percentage = (count / total_samples) * 100
        class_name = label_columns[class_idx] if class_idx < len(label_columns) else f"Class_{class_idx}"
        print(f"{class_name}: {count} samples ({percentage:.2f}%)")

    # Calculate imbalance metrics
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count

    print(f"\nImbalance Metrics:")
    print(f"Max samples: {max_count}")
    print(f"Min samples: {min_count}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    return {
        'class_counts': dict(zip([label_columns[i] for i in unique_classes], counts)),
        'imbalance_ratio': imbalance_ratio,
        'total_samples': total_samples
    }


# Example usage functions to add to your train.py

def create_three_step_dataloaders(df_train, df_val, df_test, known_labels, new_order_labels,
                                  args, features_flow, features_host):
    """
    Create all necessary dataloaders for three-step training.

    Returns:
        train_loader_random: for pre-training with balanced random labels
        train_loader_original: for fine-tuning with original labels
        val_loader: for validation
        test_loader: for testing
        label_info: information about label transformations
    """
    print("=" * 60)
    print("CREATING THREE-STEP DATALOADERS")
    print("=" * 60)

    # Analyze original distribution
    print("\n1. Analyzing original training data distribution...")
    distribution_info = analyze_class_distribution(df_train, known_labels)

    # Create random label dataloader for pre-training
    print("\n2. Creating pre-training dataloader with balanced random labels...")
    train_loader_random, label_info = create_dataloader_with_mode(
        df_train, known_labels, args.batch_size, features_flow, features_host,
        mode='random', balance_strategy='uniform'
    )

    # Create original label dataloader for fine-tuning
    print("\n3. Creating fine-tuning dataloader with original labels...")
    train_loader_original, _ = create_dataloader_with_mode(
        df_train, known_labels, args.batch_size, features_flow, features_host,
        mode='original'
    )

    # Create validation and test loaders (always use original labels)
    print("\n4. Creating validation and test dataloaders...")
    val_loader = create_dataloader(df_val, known_labels, args.batch_size, features_flow, features_host)
    test_loader = create_dataloader(df_test, new_order_labels, args.batch_size, features_flow, features_host)

    print("\n✅ All dataloaders created successfully!")

    return train_loader_random, train_loader_original, val_loader, test_loader, label_info


def print_three_step_summary(label_info, distribution_info):
    """
    Print a summary of the three-step approach setup.
    """
    print("\n" + "=" * 80)
    print("THREE-STEP TRAINING SUMMARY")
    print("=" * 80)

    print(f"\n📊 Original Data Characteristics:")
    print(f"   • Total samples: {distribution_info['total_samples']}")
    print(f"   • Imbalance ratio: {distribution_info['imbalance_ratio']:.2f}")
    print(f"   • Class distribution: {distribution_info['class_counts']}")

    if label_info:
        print(f"\n🔄 Random Label Generation:")
        print(f"   • Strategy: Uniform distribution")
        print(f"   • Original max/min ratio: {label_info['balance_improvement']['original_max_ratio']:.2f}")
        print(f"   • Random max/min ratio: {label_info['balance_improvement']['random_max_ratio']:.2f}")
        print(
            f"   • Improvement factor: {label_info['balance_improvement']['original_max_ratio'] / label_info['balance_improvement']['random_max_ratio']:.2f}x")

    print(f"\n🎯 Training Strategy:")
    print(f"   • Step 1-2: Pre-train with balanced random labels")
    print(f"   • Step 3: Fine-tune with original labels + class weights")
    print(f"   • Expected benefit: Reduced bias toward dominant classes")

    print("=" * 80)
