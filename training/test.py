import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score


@torch.no_grad()
def test(model, device, batch_size, loader, compute_loss=True):
    model.eval()
    y_pred = []
    y_true = []
    n_subgraphs = 0
    total_loss = 0

    for batch in loader:
        batch.to(device)
        # out = model(batch.x_dict, batch.edge_index_dict)
        out = model(batch)
        if compute_loss:
            # loss = F.cross_entropy(out, batch['flow'].y.float())
            loss = F.cross_entropy(out, batch['flow'].y.argmax(dim=1).long())
            total_loss += float(loss) * batch_size

        y_pred.append(out.argmax(dim=1))
        y_true.append(batch['flow'].y.argmax(dim=1))
        n_subgraphs += batch_size

    y_pred = torch.cat(y_pred).cpu()
    y_true = torch.cat(y_true).cpu()
    f1score = f1_score(y_true, y_pred, average='macro')

    return total_loss / n_subgraphs, f1score, y_pred, y_true
