import torch

import numpy as np
import pandas as pd

from tqdm import tqdm


def apply_perturbation(train_loader, df_train):
    # Get original edge index from test set
    original_edge_index = train_loader.dataset[0]['host', 'flow'].edge_index
    # Get target nodes (non-benign nodes in test set)
    target_nodes = np.where(df_train[['bruteForce', 'dos', 'pingScan', 'portScan']].sum(axis=1) > 0)[0]
    # Get total number of nodes
    num_nodes = original_edge_index.max().item() + 1
    # Apply attack
    perturbed_edge_index = perturb_graph(edge_index=original_edge_index,
                                         target_nodes=target_nodes,
                                         budget=3,
                                         num_nodes=num_nodes)
    # Update test dataset with perturbed edges
    train_loader.dataset[0]['host', 'flow'].edge_index = perturbed_edge_index
    train_loader.dataset[0]['flow', 'host'].edge_index = perturbed_edge_index.flip(0)  # Ensure symmetry

    return train_loader


def perturb_graph(edge_index, target_nodes, budget, num_nodes):
    """
    Efficiently perturbs the edge list by inserting/deleting edges for target nodes.

    Parameters:
        edge_index (torch.Tensor): The original edge index (shape: [2, num_edges]).
        target_nodes (list): Indices of the target nodes to attack.
        budget (int): Maximum number of edges to modify per node.
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        torch.Tensor: The perturbed edge index.
    """
    edge_array = edge_index.cpu().numpy().T  # Convert to NumPy for fast processing
    edge_set = set(map(tuple, edge_array))  # Store edges as a set for quick lookups

    for node in tqdm(target_nodes, desc="Perturbing edges", unit="node"):
        modified_edges = 0

        while modified_edges < budget:
            batch_size = min(budget - modified_edges, 10)  # Process up to 10 edges at a time

            # Generate candidate edges
            new_neighbors = np.random.randint(0, num_nodes, size=batch_size)  # Ensure valid node range
            new_edges = [(node, n) for n in new_neighbors if node != n and 0 <= n < num_nodes]

            # Remove duplicates & already existing edges
            new_edges = [e for e in new_edges if e not in edge_set]
            edge_set.update(new_edges)  # Add new edges
            modified_edges += len(new_edges)

            # Randomly delete edges (ensure valid deletions)
            existing_edges = [e for e in edge_set if node in e and e[0] != e[1]]
            if existing_edges:
                edges_to_remove = np.random.choice(len(existing_edges), size=min(batch_size, len(existing_edges)),
                                                   replace=False)
                for idx in edges_to_remove:
                    if existing_edges[idx] in edge_set:
                        edge_set.remove(existing_edges[idx])
                        modified_edges += 1

    # Convert back to tensor format
    perturbed_edge_index = torch.tensor(list(edge_set), dtype=torch.long).T.to(edge_index.device)

    # Ensure correct shape and valid indices
    if perturbed_edge_index.shape[0] != 2:
        raise ValueError(f"Invalid edge_index shape: {perturbed_edge_index.shape}, expected [2, num_edges]")

    if perturbed_edge_index.max() >= num_nodes:
        raise ValueError(
            f"Edge index contains out-of-bounds nodes: max={perturbed_edge_index.max()}, num_nodes={num_nodes}")

    return perturbed_edge_index
