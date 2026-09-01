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
