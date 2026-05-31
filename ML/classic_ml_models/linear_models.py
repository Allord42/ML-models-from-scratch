import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.1, C=1, num_iter=1000):
        self.lr = lr
        self.C = C
        self.num_iter = num_iter
        self.losses = []

    def sigmoid_usual(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid(self, z):
        z = np.array(z, dtype=float)

        positive = z >= 0
        negative = ~positive

        out = np.empty_like(z, dtype=float)

        out[positive] = 1 / (1 + np.exp(-z[positive]))
        out[negative] = np.exp(z[negative]) / (1 + np.exp(z[negative]))

        return out

    def loss(self, probs, y, n):
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)

        loss = -(1 / n) * np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))

        return loss

    def gradients(self, n, X, y, logits, metrics="neg_log_loss"):
        if metrics == "neg_log_loss":
            return {
                "weights": (1 / n) * np.matmul(X.T, (logits - y))
                + (1 / self.C) * self.weights,
                "bias": (1 / n) * sum(logits - y),
            }

    def params_renew(self, weights, bias, gradient_weights, gradient_bias):
        weights = weights.copy()  # to not change original weights

        weights -= np.dot(self.lr, gradient_weights)
        bias -= np.dot(self.lr, gradient_bias)
        return weights, bias

    def fit(self, X, y, epsilon=1e-6):
        prev_loss = float("inf")
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        n = X.shape[0]

        for i in range(self.num_iter):
            logits = np.matmul(X, self.weights) + self.bias
            probs = self.sigmoid(logits)

            grads = self.gradients(n, X, y, probs)
            gradient_weights = grads["weights"]
            gradient_bias = grads["bias"]

            new_weights, new_bias = self.params_renew(
                self.weights, self.bias, gradient_weights, gradient_bias
            )

            loss = self.loss(probs, y, n)
            self.losses.append(loss)
            if abs(prev_loss - loss) < epsilon:
                print(f"Stopped on iteration {i}")
                break
            self.weights, self.bias = new_weights, new_bias
            prev_loss = loss

    def predict(self, X_test):
        probabilities = self.sigmoid(self.bias + np.dot(X_test, self.weights))
        classes = (probabilities >= 0.5).astype(int)
        return classes

    def predict_proba(self, X_test):
        p1 = self.sigmoid(self.bias + np.dot(X_test, self.weights))
        p0 = 1 - p1
        return np.vstack([p0, p1]).T


# Learning linear regression models using GD, while sklearn finds analytical solution
#    TODO: ADD SGD and Batch SGD and optionally analitical solving for regression


class Regression:
    def __init__(self, lr=0.001, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = np.array([])
        self.bias = 0
        self.losses = []

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def params_renew(self, weights, bias, gradient_weights, gradient_bias):
        weights = weights.copy()  # to not change original weights

        weights -= np.dot(self.lr, gradient_weights)
        bias -= np.dot(self.lr, gradient_bias)
        return weights, bias

    def fit(self, X, y, epsilon=0.1):
        prev_loss = float("inf")
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        n = X.shape[0]

        for i in range(self.num_iter):
            preds = np.matmul(X, self.weights) + self.bias

            grads = self.gradients(n, X, y, preds)
            gradient_weights = grads["weights"]
            gradient_bias = grads["bias"]

            new_weights, new_bias = self.params_renew(
                self.weights, self.bias, gradient_weights, gradient_bias
            )

            loss = self.loss(preds, y, n)
            self.losses.append(loss)
            if abs(prev_loss - loss) < epsilon:
                print(f"Stopped on iteration {i}")
                break
            self.weights, self.bias = new_weights, new_bias
            prev_loss = loss

    def predict(self, X_test):
        predictions = self.bias + np.dot(X_test, self.weights)
        return predictions


class LinearRegression(Regression):
    def __init__(self, lr=0.001, num_iter=1000):
        super().__init__(lr=lr, num_iter=num_iter)

    def __repr__(self):
        return f"My {self.__class__.__name__}"

    def __str__(self):
        return f"My {self.__class__.__name__}"

    def gradients(self, n, X, y, preds, metrics="mse"):
        if metrics == "mse":
            return {
                "weights": (2 / n) * np.matmul(X.T, (preds - y)),
                "bias": (2 / n) * sum(preds - y),
            }

    def loss(self, preds, y, n, loss="mse"):
        if loss == "mse":
            loss = (1 / n) * np.sum((preds - y) ** 2)
        return loss


class Lasso(Regression):
    def __init__(self, lr=0.001, alpha=1, num_iter=1000):
        super().__init__(lr=lr, num_iter=num_iter)
        self.alpha = alpha

    def gradients(self, n, X, y, preds, metrics="mse"):
        if metrics == "mse":
            return {
                "weights": (2 / n) * np.matmul(X.T, (preds - y))
                + self.alpha * np.sign(self.weights),  # l1 weights gradient
                "bias": (2 / n) * sum(preds - y),
            }

    def loss(self, preds, y, n, loss="mse"):
        if loss == "mse":
            loss = (1 / n) * np.sum((preds - y) ** 2) + self.alpha * np.sum(
                np.abs(self.weights)
            )  # l1 loss
        return loss


class Ridge(Regression):
    def __init__(self, lr=0.001, alpha=1, num_iter=1000):
        super().__init__(lr=lr, num_iter=num_iter)
        self.alpha = alpha

    def gradients(self, n, X, y, preds, metrics="mse"):
        if metrics == "mse":
            return {
                "weights": (2 / n) * np.matmul(X.T, (preds - y))
                + 2 * self.alpha * self.weights,  # l2 weights gradient
                "bias": (2 / n) * sum(preds - y),
            }

    def loss(self, preds, y, n, loss="mse"):
        if loss == "mse":
            loss = (1 / n) * np.sum((preds - y) ** 2) + self.alpha * np.sum(
                self.weights**2
            )  # l2 loss
        return loss


class ElasticNet(Regression):
    def __init__(self, lr=0.001, alpha=1, l1_ratio=0.5, num_iter=1000):
        super().__init__(lr=lr, num_iter=num_iter)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def gradients(self, n, X, y, preds, metrics="mse"):
        if metrics == "mse":
            return {
                "weights": (2 / n) * np.matmul(X.T, (preds - y))
                + self.alpha
                * (
                    self.l1_ratio * np.sign(self.weights)  # l1 weights gradient
                    + 2 * self.alpha * self.weights
                ),  # l2 weights gradient
                "bias": (2 / n) * sum(preds - y),
            }

    def loss(self, preds, y, n, loss="mse"):
        if loss == "mse":
            loss = (1 / n) * np.sum((preds - y) ** 2) + self.alpha * (
                self.l1_ratio
                * np.sum(
                    np.abs(self.weights)  # l1 loss
                    + (1 - self.l1_ratio) * np.sum(self.weights**2)
                )
            )  # l2 loss
        return loss


class SVM(Regression):
    def __init__(self, lr=0.001, C=1, num_iter=1000):
        super().__init__(lr=lr, num_iter=num_iter)
        self.C = C

    def gradients(self, X, y, preds, metrics="hinge loss'"):
        if metrics == "hinge loss'":
            if preds < 1:
                return {
                    "weights": self.weights - self.C * (y * X),
                    "bias": -sum(self.C * y),
                }
            return {"weights": self.weights, "bias": 0}

    def loss(self, preds, y, loss="hinge loss"):
        if loss == "hinge loss":
            if preds >= 1:
                loss = 0.5 * np.linalg.norm(self.weights)
            else:
                loss = 0.5 * np.linalg.norm(self.weights) + self.C * y * preds
        return loss


class SVM(Regression):
    def __init__(self, lr=0.001, C=1, num_iter=1000):
        super().__init__(lr=lr, num_iter=num_iter)
        self.C = C

    def gradients(self, n, X, y, preds, metrics="hinge"):
        # y must be -1 or +1
        if metrics == "hinge":
            margins = y * preds

            grad_w = np.zeros_like(self.weights)
            grad_b = 0

            for i in range(n):
                if margins[i] >= 1:
                    grad_w += self.weights  # only regularization
                    grad_b += 0
                else:
                    grad_w += self.weights - self.C * y[i] * X[i]  # hinge loss active
                    grad_b += -self.C * y[i]

            grad_w /= n
            grad_b /= n

            return {"weights": grad_w, "bias": grad_b}

    def loss(self, preds, y, n, loss="hinge"):
        if loss == "hinge":
            margins = 1 - y * preds
            hinge = np.maximum(0, margins)
            return 0.5 * np.sum(self.weights**2) + self.C * np.sum(hinge)
