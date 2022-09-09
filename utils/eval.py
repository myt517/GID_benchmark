import numpy as np
import torch
from pytorch_lightning.metrics import Metric

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mapping, w = compute_best_mapping(y_true, y_pred)
    return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size

def cluster_F1(y_true, y_pred):
    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred_aligned = np.array([map_[idx] for idx in y_pred])
    F1_score = f1_score(y_true, y_pred_aligned, average='weighted')
    return F1_score

def cluster_precision(y_true, y_pred):
    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred_aligned = np.array([map_[idx] for idx in y_pred])
    precision = precision_score(y_true, y_pred_aligned, average='weighted')
    return precision

def cluster_recall(y_true, y_pred):
    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred_aligned = np.array([map_[idx] for idx in y_pred])
    recall = recall_score(y_true, y_pred_aligned, average='weighted')
    return recall

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def compute_best_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w))), w


class ClusterMetrics(Metric):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.add_state("preds", default=[])
        self.add_state("targets", default=[])

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds, dim=-1)
        targets = torch.cat(self.targets)
        targets -= targets.min()
        acc, nmi, ari, f1, precision, recall = [], [], [], [], [], []
        for head in range(self.num_heads):
            t = targets.cpu().numpy()
            p = preds[head].cpu().numpy()
            acc.append(torch.tensor(cluster_acc(t, p), device=preds.device))
            nmi.append(torch.tensor(nmi_score(t, p), device=preds.device))
            ari.append(torch.tensor(ari_score(t, p), device=preds.device))
            f1.append(torch.tensor(cluster_F1(t, p), device=preds.device))
            precision.append(torch.tensor(cluster_precision(t, p), device=preds.device))
            recall.append(torch.tensor(cluster_recall(t, p), device=preds.device))

        return {"acc": acc, "nmi": nmi, "ari": ari, "f1": f1, "pre": precision, "rec": recall}


class classifyMetrics(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[])
        self.add_state("targets", default=[])

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds, dim=-1)
        targets = torch.cat(self.targets)
        targets -= targets.min()
        acc, f1, precision, recall = [], [], [], []

        t = targets.cpu().numpy()
        p = preds.cpu().numpy()
        acc.append(torch.tensor(accuracy_score(t, p), device=preds.device))
        f1.append(torch.tensor(f1_score(t, p, average='weighted'), device=preds.device))
        precision.append(torch.tensor(precision_score(t, p, average='weighted'), device=preds.device))
        recall.append(torch.tensor(recall_score(t, p, average='weighted'), device=preds.device))

        return {"acc": acc, "f1": f1, "pre": precision, "rec": recall}
