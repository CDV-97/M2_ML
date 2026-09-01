class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def gini(self, y):
        if len(y) == 0:
            return 0

        counts = {}

        for label in y:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1

        impurity = 1

        for label in counts:
            probability = counts[label] / len(y)
            impurity -= probability ** 2

        return impurity

    def split(self, X, y, feature, threshold):
        X_left = []
        y_left = []

        X_right = []
        y_right = []

        for i in range(len(X)):
            if X[i][feature] <= threshold:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        return X_left, y_left, X_right, y_right

    def best_split(self, X, y):
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        num_features = len(X[0])

        for feature in range(num_features):
            thresholds = []

            for row in X:
                value = row[feature]
                if value not in thresholds:
                    thresholds.append(value)

            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split(
                    X, y, feature, threshold
                )

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                total = len(y)

                weighted_gini = (
                    len(y_left) / total * self.gini(y_left)
                    + len(y_right) / total * self.gini(y_right)
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
