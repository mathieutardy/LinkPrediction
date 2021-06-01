# Packages for model training
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np


class ModelTraining:
    @staticmethod
    def prepare_data_for_training(df, metrics_to_keep):
        """
        Performs train test split of dataset for training and validation

        Parameters
        ----------
        df : dataframe
          dataframe of node comparison with computed metrics.
        metrics_to_keep: list
          list of columns name that you want to use in the training process.
          All features are strings.

        Returns
        -------
        4 numpy arrays
          Classic X_train, X_test, y_train and y_test

        """
        df = df.set_index(["node1", "node2"])
        X = df[metrics_to_keep].values
        y = df["link"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=1
        )

        return X_train, X_test, y_train, y_test

    @staticmethod
    def ensemble_classifier_training(X_train, X_test, y_train, y_test):

        """
        Trains a list of classifiers.
        We compute AUC score, F1 and accuracy score of each models.
        Returns a summary of these results in a dataframe.

        """

        classifiers = [
            KNeighborsClassifier(3),
            LinearSVC(),
            LogisticRegression(),
            RidgeClassifier(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            XGBClassifier(),
        ]

        lst = []

        for clf in classifiers:
            clf.fit(X_train, y_train)
            name = clf.__class__.__name__
            y_pred = clf.predict(X_test)
            f1 = "{:.1%}".format(f1_score(y_test, y_pred, average="macro"))
            acc = "{:.1%}".format(accuracy_score(y_test, y_pred))
            auc = "{:.1%}".format(roc_auc_score(y_test, y_pred))
            lst.append([name, f1, acc, auc])

        results = pd.DataFrame(lst, columns=["Classifier", "F1", "Accuracy", "AUC"])
        results = results.sort_values(by="F1", ascending=False)

        return results
