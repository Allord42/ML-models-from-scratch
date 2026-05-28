def mean_squared_error(y_test, predictions):
    if len(y_test) != len(predictions):
        raise ValueError("Length of y_test and predictions must match")

    return (1 / len(y_test)) * sum((y_test - predictions) ** 2)

def log_losss(y_test, probabilities):
    if len(y_test) != len(probabilities):
        raise ValueError("Length of y_test and probabilities must match")
    
    probs_1 = probabilities[:, 1]

    return (1 / len(y_test)) * sum(- (y_test * np.log(probs_1) + (1 - y_test) * np.log(1 - probs_1)))