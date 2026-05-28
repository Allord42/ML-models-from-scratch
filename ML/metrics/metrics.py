import numpy as np


def mean_squared_error(y_test, predictions):
    if len(y_test) != len(predictions):
        raise ValueError("Length of y_test and predictions must match")

    return (1 / len(y_test)) * sum((y_test - predictions) ** 2)


def log_losss(y_test, probabilities):
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
    FPR = TN / (FP + TN)

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
