"""Classify Iris flowers with a scaled logistic regression model."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run():
    """Train and evaluate the classifier, returning its test accuracy."""
    data = load_iris()
    features_train, features_test, labels_train, labels_test = train_test_split(
        data.data,
        data.target,
        random_state=42,
        stratify=data.target,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features_train, labels_train)
    return accuracy_score(labels_test, model.predict(features_test))


if __name__ == "__main__":
    print(f"Iris classification accuracy: {run():.3f}")
