# MLLab

Small, reproducible machine-learning projects that demonstrate common workflows.
The examples use datasets bundled with scikit-learn or generated in code, so no
external data downloads are needed.

## Projects

| Project | Task | Example |
| --- | --- | --- |
| Iris classifier | Multiclass classification | Scales flower measurements and predicts species with logistic regression |
| Diabetes predictor | Regression | Predicts a quantitative disease-progression measure with linear regression |
| Blob clustering | Unsupervised learning | Groups generated points with K-means and measures silhouette score |

## Run an example

Requires Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python -m projects.iris_classifier
python -m projects.diabetes_regression
python -m projects.blob_clustering
```

Each command prints a reproducible evaluation metric. To run the tests:

```bash
python -m unittest discover -s tests
```
