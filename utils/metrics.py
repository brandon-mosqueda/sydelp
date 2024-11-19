import numpy as np

from sklearn.metrics import recall_score, f1_score as _f1_score
from utils.utils import Float, FloatArray, IntArray, Int, NumArray


def label_flipping_success_rate(y_true: IntArray,
                                y_pred: IntArray,
                                source: Int,
                                target: Int) -> Float:
    if y_true.size != y_pred.size:
        raise ValueError('y_true and y_pred must have the same length')

    source_indices: IntArray = np.where(y_true == source)[0]

    if not y_true.size or not source_indices.size:
        return 0

    success_rate: int = np.sum(y_pred[source_indices] == target)

    return success_rate / len(source_indices)


def label_recall(y_true: IntArray, y_pred: IntArray, label: Int) -> Float:
    if y_true.size != y_pred.size:
        raise ValueError('y_true and y_pred must have the same length')

    if not y_true.size:
        return 0

    recalls: FloatArray = np.array(recall_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(label + 1),
        zero_division=0
    ))

    return recalls[label]


def f1_score(y_true: IntArray, y_pred: IntArray) -> Float:
    return float(_f1_score(y_true, y_pred, zero_division=0))


def cos_similarity(x: NumArray, y: NumArray) -> Float:
    res: np.float_ = (x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y))

    return np.float_(min(res, 1))
