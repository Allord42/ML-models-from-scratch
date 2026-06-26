import numpy as np
from dataclasses import dataclass
from graphviz import Digraph
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
)  # for gridsearch


@dataclass
class Node:
    feature: str = None
    threshold: float = None
    left: "Node" = None
    right: "Node" = None
    prediction: int = None  # in the final leaf
    probability: dict = None  # probability of classes in the leaf


class Tree:
    def __init__(
        self,
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        round_digits=4,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.round_digits = round_digits

    @staticmethod
    def gini(pos, neg):
        "Gets number of pos and neg class in the bunch"

        total = pos + neg
        if total == 0:
            return 0

        p_pos = pos / total
        p_neg = neg / total

        H = 1
        H -= p_pos**2 + p_neg**2
        return H

    @staticmethod
    def entropy(pos, neg):
        "Gets number of pos and neg class in the bunch"

        total = pos + neg
        if total == 0:
            return 0

        p_pos = pos / total
        p_neg = neg / total

        H = 0
        if p_pos > 0:
            H -= p_pos * np.log2(p_pos)
        if p_neg > 0:
            H -= p_neg * np.log2(p_neg)
        return H

    @staticmethod
    def mse(y):
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def class_prediction_in_leaf(self, y):
        if isinstance(self, DecisionTreeClassifier):
            values, counts = np.unique(y, return_counts=True)
            most_frequent = values[counts.argmax()]
            return most_frequent
        if isinstance(self, DecisionTreeRegressor):
            return np.round(sum(y) / len(y), self.round_digits)

    def visualize_tree(self, node, dot=None, parent=None, edge_label=""):
        if dot is None:
            dot = Digraph()
            dot.attr("node", shape="box", style="filled")

        if node.prediction is not None:
            label = f"Leaf\npredict={node.prediction}"
            color = (
                "lightgreen"
                if isinstance(self, DecisionTreeClassifier)
                else "lightyellow"
            )

        else:
            label = f"{node.feature} <= {node.threshold:.2f}"
            color = "lightblue"

        node_id = str(id(node))
        dot.node(node_id, label, fillcolor=color)

        if parent:
            dot.edge(parent, node_id, label=edge_label)

        if node.left:
            self.visualize_tree(node.left, dot, node_id, "left")
        if node.right:
            self.visualize_tree(node.right, dot, node_id, "right")

        return dot

    def check_stop_conditions(self, N, y, depth):
        if len(set(y)) == 1:  # all labels are the same in the leaf
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if N < self.min_samples_split:
            return True
        return False

    def build_tree(self, X, y, depth, current_impurity=float("inf")):
        N = len(X)
        y = np.array(y)
        if self.check_stop_conditions(N, y, depth):
            if isinstance(self, DecisionTreeClassifier):
                return Node(
                    prediction=self.class_prediction_in_leaf(y),
                    probability=self.probability_prediction_in_leaf(y),
                )
            elif isinstance(self, DecisionTreeRegressor):
                return Node(prediction=self.class_prediction_in_leaf(y))
        else:
            min_H = current_impurity
            best_col_to_split, best_threshold = None, None
            best_left_idx, best_right_idx = None, None

            for col_name in X.columns:
                # sorting values in column for finding thresholds between them
                col = np.array(X[col_name])
                idx = np.argsort(col)
                X_sorted = col[idx]

                for i in range(1, len(X)):
                    if X_sorted[i] == X_sorted[i - 1]:
                        continue

                    threshold = (X_sorted[i] + X_sorted[i - 1]) / 2

                    left_idx = idx[:i]
                    right_idx = idx[i:]

                    y_left = y[left_idx]
                    y_right = y[right_idx]

                    len_left, len_right = len(y_left), len(y_right)
                    pos_in_left, pos_in_right = sum(y_left), sum(y_right)
                    neg_in_left, neg_in_right = (
                        len_left - pos_in_left,
                        len_right - pos_in_right,
                    )

                    if isinstance(self, DecisionTreeClassifier):
                        impurity = getattr(Tree, self.criterion)
                        G_left = impurity(pos_in_left, neg_in_left)
                        G_right = impurity(pos_in_right, neg_in_right)
                    else:  # DecisionTreeRegressor
                        G_left = Tree.mse(y_left)
                        G_right = Tree.mse(y_right)

                    H_after_split = (len_left / N) * G_left + (len_right / N) * G_right

                    if (
                        H_after_split < min_H
                        and len_left >= self.min_samples_leaf
                        and len_right >= self.min_samples_leaf
                    ):
                        min_H = H_after_split
                        best_col_to_split = col_name
                        best_threshold = threshold
                        best_left_idx, best_right_idx = left_idx, right_idx

        if best_col_to_split is None or min_H == current_impurity:  # early stopping
            if isinstance(self, DecisionTreeClassifier):
                return Node(
                    prediction=self.class_prediction_in_leaf(y),
                    probability=self.probability_prediction_in_leaf(y),
                )
            elif isinstance(self, DecisionTreeRegressor):
                return Node(prediction=self.class_prediction_in_leaf(y))

        left = self.build_tree(X.iloc[best_left_idx], y[best_left_idx], depth + 1)
        right = self.build_tree(X.iloc[best_right_idx], y[best_right_idx], depth + 1)
        node = Node(best_col_to_split, best_threshold, left, right)
        return node

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class DecisionTreeClassifier(Tree, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        round_digits=4,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            round_digits=round_digits,
        )

    def probability_prediction_in_leaf(self, y):
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return dict(zip(values, np.round(probs, self.round_digits)))

    def predict_one(self, x, node=None, proba=None):
        if node is None:
            node = self.root
        if node.prediction is not None:
            if proba == "proba":
                return node.probability
            else:
                return node.prediction
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left, proba)
        else:
            return self.predict_one(x, node.right, proba)

    def fit(self, X_train, y_train, depth=1):
        self.classes_ = np.unique(y_train)
        self.root = self.build_tree(X_train, y_train, depth)
        return self.root

    def predict(self, X):
        return np.array([self.predict_one(row) for _, row in X.iterrows()])

    def predict_proba(self, X):
        probs = []
        for _, row in X.iterrows():
            leaf_prob = self.predict_one(row, proba="proba")
            row_probs = [leaf_prob.get(c, 0.0) for c in self.classes_]
            probs.append(row_probs)
        return np.array(probs)


class DecisionTreeRegressor(Tree, BaseEstimator, RegressorMixin):
    def __init__(
        self,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        round_digits=4,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            round_digits=round_digits,
        )

    def predict_one(self, x, node=None, proba=None):
        if node is None:
            node = self.root
        if node.prediction is not None:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left, proba)
        else:
            return self.predict_one(x, node.right, proba)

    def fit(self, X_train, y_train, depth=1):
        self.root = self.build_tree(X_train, y_train, depth)
        return self.root

    def predict(self, X):
        return np.array([self.predict_one(row) for _, row in X.iterrows()])

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
