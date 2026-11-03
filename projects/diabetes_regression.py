"""Predict diabetes progression with a linear regression model."""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def run():
    """Train and evaluate the regressor, returning its test R-squared."""
    data = load_diabetes()
    features_train, features_test, labels_train, labels_test = train_test_split(
        data.data,
        data.target,
        random_state=42,
    )
    model = LinearRegression()
    model.fit(features_train, labels_train)
    return r2_score(labels_test, model.predict(features_test))


if __name__ == "__main__":
    print(f"Diabetes regression R-squared: {run():.3f}")
