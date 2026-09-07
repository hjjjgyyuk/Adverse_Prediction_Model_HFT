from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

import numpy as np
import pandas as pd


def train_model(X_train, y_train):

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\nFeature importance:")
    print(importance)

    return model


def evaluate_model(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    print("\nProbability statistics:")
    print("Minimum:", probabilities.min())
    print("Maximum:", probabilities.max())
    print("Mean:", probabilities.mean())

    print("\nProbability percentiles:")
    print(
        np.percentile(
            probabilities,
            [50, 75, 90, 95, 99, 99.5, 99.9]
        )
    )

    thresholds = [
        0.02,
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20
    ]

    print("\nThreshold evaluation:")
    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'Predicted':<12}"
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        predicted = predictions.sum()

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.3f}"
            f"{recall:<12.3f}"
            f"{predicted:<12}"
        )

    threshold = 0.05

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print(f"\nClassification Report at threshold {threshold}:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return probabilities