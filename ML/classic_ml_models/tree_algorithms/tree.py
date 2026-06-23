import numpy as np
from dataclasses import dataclass
from graphviz import Digraph


@dataclass
class Node:
    feature: str = None
    threshold: float = None
    left: "Node" = None
    right: "Node" = None
    prediction: int = None  # in final leaf


@dataclass
class Tree:
    """
    Classifier Tree with classes 0 and 1
    """

    criterion: str = "entropy"
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

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

    def visualize_tree(self, node, dot=None, parent=None, edge_label=""):
        if dot is None:
            dot = Digraph()
            dot.attr("node", shape="box", style="filled")

        if node.prediction is not None:
            if node.prediction == 1:
                label = "Leaf\npredict=1"
                color = "lightgreen"
            else:
                label = "Leaf\npredict=0"
                color = "lightcoral"
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

    def _prediction_in_leaf(self, y):
        if sum(y) > len(y) // 2:
            return 1
        return 0

    def _check_stop_conditions(self, N, y, depth):
        if len(set(y)) == 1:  # all labels are the same in the leaf
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if N < self.min_samples_split:
            return True
        return False

    def build_tree(self, X, y, depth):
        N = len(X)
        if self._check_stop_conditions(N, y, depth):
            return Node(prediction=self._prediction_in_leaf(y))
        else:
            min_H = float("inf")
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

                    if self.criterion == "gini":
                        G_left = Tree.gini(pos_in_left, neg_in_left)
                        G_right = Tree.gini(pos_in_right, neg_in_right)
                    else:
                        G_left = Tree.entropy(pos_in_left, neg_in_left)
                        G_right = Tree.entropy(pos_in_right, neg_in_right)

                    H_after_split = (len_left / N) * G_left + (len_right / N) * G_right

                    if H_after_split < min_H:
                        min_H = H_after_split
                        best_col_to_split = col_name
                        best_threshold = threshold
                        best_left_idx, best_right_idx = left_idx, right_idx

        if best_col_to_split is None:
            return Node(prediction=self._prediction_in_leaf(y))

        left = self.build_tree(X.iloc[best_left_idx], y[best_left_idx], depth + 1)
        right = self.build_tree(X.iloc[best_right_idx], y[best_right_idx], depth + 1)
        node = Node(best_col_to_split, best_threshold, left, right)
        return node

    def predict_one(self, x, node=None):
        if node is None:
            node = self.root
        if node.prediction is not None:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def fit(self, X_train, y_train, depth=1):
        self.root = self.build_tree(X_train, y_train, depth)
        return self.root

    def predict(self, X):
        return np.array([self.predict_one(row) for _, row in X.iterrows()])
