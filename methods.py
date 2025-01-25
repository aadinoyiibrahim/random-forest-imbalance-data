"""
Project: Random forest on imbalanced dataset

this file contains the implementation of the robust decision tree and robust random forest

Author: Abdullahi A. Ibrahim
date: 21-01-2025
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RobustDecisionTree(DecisionTreeClassifier):
    def __init__(self, epsilon=0.1, **kwargs):
        """
        A decision tree with perturbation
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def robust_gini(self, X, y):
        """compute gini impurity"""
        perturbed_X = X + np.random.uniform(-self.epsilon, self.epsilon, X.shape)
        return self.weighted_gini(y)

    def fit(self, X, y, sample_weight=None):
        """override fit with perturbation"""
        X_perturbed = X + np.random.uniform(-self.epsilon, self.epsilon, X.shape)
        return super().fit(X_perturbed, y, sample_weight)


class RobustRandomForest(RandomForestClassifier):
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.train_accuracies = []
        self.val_accuracies = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        def check_classification_targets(y):
            """
            Validate the target array
            """
            if not np.issubdtype(y.dtype, np.integer):
                raise ValueError("Classification targets must be integers.")
            if len(np.unique(y)) < 2:
                raise ValueError(
                    "There must be at least two classes for classification."
                )

        check_classification_targets(y_train)
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        self.n_outputs_ = 1

        # Fitting trees
        self.estimators_ = []
        for i in range(self.n_estimators):
            tree = RobustDecisionTree(
                epsilon=self.epsilon,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            tree.fit(X_train, y_train)
            self.estimators_.append(tree)

            y_train_pred = self.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            self.train_accuracies.append(train_accuracy)

            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                self.val_accuracies.append(val_accuracy)

        return self
