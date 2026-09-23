"""Safe synthetic round-trip check for the lightweight model artifact path."""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_sklearn_prediction_survives_joblib_round_trip(tmp_path) -> None:
    examples = [
        "invented card question alpha",
        "invented card question beta",
        "invented transfer question alpha",
        "invented transfer question beta",
    ]
    labels = ["card", "card", "transfer", "transfer"]
    model = Pipeline(
        [
            ("features", TfidfVectorizer()),
            ("classifier", LogisticRegression(random_state=7)),
        ]
    ).fit(examples, labels)
    artifact = tmp_path / "synthetic.joblib"
    joblib.dump(model, artifact)
    restored = joblib.load(artifact)
    assert restored.predict(["invented card question"])[0] == "card"
