import numpy as np
from itertools import combinations
from sklearn.metrics import roc_auc_score


def multilabel_macro_auc(y_true, y_pred):
    """
    Compute the macro AUC for multilabel classification.
    """
    return roc_auc_score(y_true, y_pred, average='macro')


def multilabel_weighted_macro_auc(y_true, y_pred, likelihood=None):
    """
    Compute the weighted macro AUC for multilabel classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    likelihood = np.array(likelihood)
    auc_j_sum = 0
    m, l = y_true.shape
    for j in range(l):
        s_j = 0
        y_j_total = 0
        
        for r1, r2 in combinations(range(m), 2):
            if y_true[r1, j] != y_true[r2, j]:
                lh = likelihood[r1, j] + likelihood[r2, j]
                y_j_total += lh
                pos, neg = (r1, r2) if y_true[r1, j] > y_true[r2, j] else (r2, r1)
                if y_pred[pos, j] > y_pred[neg, j]:
                    s_j += lh
                # Same probs (draw)
                elif y_pred[pos, j] == y_pred[neg, j]:
                    s_j += lh/2
        if y_j_total == 0:  # Avoid division by zero
            continue
        auc_j_sum += s_j / y_j_total
    return 1 / l * auc_j_sum


def multilabel_instance_auc(y_true, y_pred):
    """
    Compute the instance AUC for multilabel classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auc_i_sum = 0
    m, l = y_true.shape
    for i in range(m):
        s_i = 0
        y_i_plus = sum(y_true[i, :])
        y_i_minus = l - y_i_plus
        y_i_total = y_i_plus * y_i_minus
        for c1, c2 in combinations(range(l), 2):
            if y_true[i, c1] != y_true[i, c2]:
                pos, neg = (c1, c2) if y_true[i, c1] > y_true[i, c2] else (c2, c1)
                if y_pred[i, pos] > y_pred[i, neg]:
                    s_i += 1
                # Same probs (draw)
                elif y_pred[i, pos] == y_pred[i, neg]:
                    s_i += 0.5
        if y_i_total == 0:  # Avoid division by zero
            continue
        auc_i_sum += s_i / y_i_total
    return 1 / m * auc_i_sum


def multilabel_weighted_instance_auc(y_true, y_pred, likelihood=None):
    """
    Compute the weighted instance AUC for multilabel classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    likelihood =  np.array(likelihood)
    auc_i_sum = 0
    m, l = y_true.shape
    for i in range(m):
        s_i = 0
        y_i_total = 0
        
        for c1, c2 in combinations(range(l), 2):
            if y_true[i, c1] != y_true[i, c2]:
                lh = likelihood[i, c1] + likelihood[i, c2]
                y_i_total += lh
                pos, neg = (c1, c2) if y_true[i, c1] > y_true[i, c2] else (c2, c1)
                if y_pred[i, pos] > y_pred[i, neg]:
                    s_i += lh
                # Same probs (draw)
                elif y_pred[i, pos] == y_pred[i, neg]:
                    s_i += lh / 2
        if y_i_total == 0:  # Avoid division by zero
            continue
        auc_i_sum += s_i / y_i_total
    return 1 / m * auc_i_sum
