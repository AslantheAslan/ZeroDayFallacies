import torch

from sklearn.covariance import EmpiricalCovariance


def compute_mahalanobis_stats(flow_embeddings, flow_labels, target_class='benign'):
    # Select embeddings for benign class
    mask = (flow_labels == target_class)
    benign_embeds = flow_embeddings[mask]

    # Fit covariance model
    cov = EmpiricalCovariance().fit(benign_embeds.cpu().numpy())
    mu = cov.location_
    sigma_inv = cov.precision_

    return torch.tensor(mu).to(flow_embeddings.device), torch.tensor(sigma_inv).to(flow_embeddings.device)


# def mahalanobis_distance(x, mu, sigma_inv):
#     diff = x - mu
#     return torch.sqrt((diff @ sigma_inv) * diff).sum(dim=1)

def mahalanobis_distance(x, mu, sigma_inv):
    diff = x - mu
    temp = diff @ sigma_inv
    mahal_squared = (temp * diff).sum(dim=1)
    mahal_squared = torch.clamp(mahal_squared, min=0.0)  # Critical: handle numerical errors
    return torch.sqrt(mahal_squared)
