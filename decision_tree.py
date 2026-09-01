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

    def most_common_label(self, y):
        counts = {}

        for label in y:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1

        most_common = None
        highest_count = 0

        for label in counts:
            if counts[label] > highest_count:
                highest_count = counts[label]
                most_common = label

        return most_common

    def grow_tree(self, X, y, depth=0):
        num_samples = len(y)
        num_classes = len(set(y))

        # Condiciones para detener el crecimiento
        if (
            depth >= self.max_depth
            or num_samples < self.min_samples_split
            or num_classes == 1
        ):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Buscar la mejor división
        best_feature, best_threshold = self.best_split(X, y)

        # Si no encontramos una división válida
        if best_feature is None:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Dividir los datos
        X_left, y_left, X_right, y_right = self.split(
            X, y, best_feature, best_threshold
        )

        # Crear las ramas recursivamente
        left_child = self.grow_tree(X_left, y_left, depth + 1)
        right_child = self.grow_tree(X_right, y_right, depth + 1)

        # Crear el nodo actual
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def fit(self, X, y):
        self.root = self.grow_tree(X, y)
