import unittest

from projects.blob_clustering import run as run_clustering
from projects.diabetes_regression import run as run_regression
from projects.iris_classifier import run as run_classifier


class ProjectTests(unittest.TestCase):
    def test_iris_classifier_accuracy(self):
        self.assertGreaterEqual(run_classifier(), 0.9)

    def test_diabetes_regression_score(self):
        self.assertGreater(run_regression(), 0.3)

    def test_blob_clustering_score(self):
        self.assertGreater(run_clustering(), 0.8)


if __name__ == "__main__":
    unittest.main()
