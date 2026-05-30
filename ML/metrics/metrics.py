import numpy as np


def mean_squared_error(y_test, predictions):
    if len(y_test) != len(predictions):
        raise ValueError("Length of y_test and predictions must match")

    return (1 / len(y_test)) * sum((y_test - predictions) ** 2)


def log_loss(y_test, probabilities):
    if len(y_test) != len(probabilities):
        raise ValueError("Length of y_test and probabilities must match")

    probs_1 = probabilities[:, 1]

    return (1 / len(y_test)) * sum(
        -(y_test * np.log(probs_1) + (1 - y_test) * np.log(1 - probs_1))
    )


def confusion_matrix(y_test, predictions):
    if len(y_test) != len(predictions):
        raise ValueError("Length of y_test and predictions must match")

    TP = TN = FP = FN = 0

    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            if y_test[i] == 0:
                TN += 1
            else:
                TP += 1
        else:
            if y_test[i] == 0:
                FP += 1
            else:
                FN += 1
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "TPR": TPR,
        "FPR": FPR,
    }


def accuracy_score(y_test, predictions):
    params = confusion_matrix(y_test, predictions)

    return (params["TP"] + params["TN"]) / (
        params["TP"] + params["FP"] + params["FN"] + params["TN"]
    )


def precision_score(y_test, predictions):
    params = confusion_matrix(y_test, predictions)

    return params["TP"] / (params["TP"] + params["FP"])


def recall_score(y_test, predictions):
    params = confusion_matrix(y_test, predictions)

    return params["TPR"]


def f1_score(y_test, predictions, beta=1):
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    return (1 + beta**2) * ((precision * recall) / (beta**2 * precision + recall))


def precision_recall_curve(y_test, probs):
    # sklearn, in opposite, returns from recall=1, pr=0 and lower thresholds
    precisions = [1]
    recalls = [0]
    thresholds = []

    p1 = probs[:, 1]
    descendning_idx = np.argsort(p1)[::-1]
    y_sorted = y_test[descendning_idx]
    p_sorted = p1[descendning_idx]

    for i in range(1, len(y_test) + 1):
        preds = np.zeros_like(y_sorted)
        preds[:i] = 1

        if i < len(y_test):
            thresholds.append(p_sorted[i])
        else:
            thresholds.append(0.0)

        precisions.append(precision_score(y_sorted, preds))
        recalls.append(recall_score(y_sorted, preds))

    return (precisions, recalls, thresholds)


def auc(x, y):
    """trapezoid integral sklearn method"""
    auc_score = 0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = (y[i - 1] + y[i]) / 2
        auc_score += dx * dy

    return auc_score


def roc_curve(y_test, probs):
    TPRs = []
    FPRs = []
    thresholds = []

    p1 = probs[:, 1]
    descending_idx = np.argsort(p1)[::-1]
    y_sorted = y_test[descending_idx]
    p_sorted = p1[descending_idx]

    for i in range(0, len(y_test) + 1):
        preds = np.zeros_like(y_sorted)
        preds[:i] = 1

        if i < len(y_test):
            thresholds.append(p_sorted[i])
        else:
            thresholds.append(0.0)

        params = confusion_matrix(y_sorted, preds)
        TPRs.append(params["TPR"])
        FPRs.append(params["FPR"])

    return TPRs, FPRs, thresholds


def roc_auc_score(FPRs, TPRs):
    return auc(FPRs, TPRs)


def gini_score(FPRs, TPRs):
    return 2 * roc_auc_score(FPRs, TPRs) - 1
