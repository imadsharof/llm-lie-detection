"""Black-box classifier: standardisation + logistic regression (Pacchiardi et al.)."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_bb_classifier(X_train, y_train, seed: int = 42):
    """The scaler is fitted on the training data only (no leakage)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    return clf.fit(X_train, y_train)


def bb_scores(clf, X):
    """Probability that the statement is true (label 1)."""
    return clf.predict_proba(X)[:, 1]