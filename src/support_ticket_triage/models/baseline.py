"""Word/character TF-IDF and multinomial logistic-regression baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from support_ticket_triage.data.split import sha256_file
from support_ticket_triage.evaluation.metrics import classification_metrics


def build_vectorizer(config: dict[str, Any]) -> FeatureUnion:
    """Create the fitted-only-on-development sparse feature union."""
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=tuple(config["word_ngram_range"]),
                    min_df=int(config["word_min_df"]),
                    max_features=int(config["word_max_features"]),
                    sublinear_tf=bool(config["sublinear_tf"]),
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(config["char_ngram_range"]),
                    min_df=int(config["char_min_df"]),
                    max_features=int(config["char_max_features"]),
                    sublinear_tf=bool(config["sublinear_tf"]),
                    lowercase=True,
                ),
            ),
        ]
    )


def _majority_metrics(development: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    counts = development["category"].value_counts()
    labels = np.array(sorted(counts.index))
    prior = np.array([counts.get(label, 0) for label in labels], dtype=float)
    prior /= prior.sum()
    probabilities = np.repeat(prior[None, :], len(validation), axis=0)
    prediction = np.repeat(counts.index[0], len(validation))
    return classification_metrics(
        validation["category"].to_numpy(), prediction, probabilities, labels
    )


def train_baseline(project_root: Path, private_root: Path) -> dict[str, Any]:
    """Tune baseline variants on validation macro F1 and save the selected model privately."""
    config = json.loads((project_root / "configs" / "baseline.json").read_text(encoding="utf-8"))
    split = pd.read_csv(private_root / "manifests" / "split_manifest_private.csv")
    development = split.loc[split["partition"] == "development"].reset_index(drop=True)
    validation = split.loc[split["partition"] == "validation"].reset_index(drop=True)
    search_rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, float, str], tuple[FeatureUnion, LogisticRegression]] = {}

    for representation in ("model_text", "redacted_text"):
        vectorizer = build_vectorizer(config)
        started = time.perf_counter()
        development_matrix = vectorizer.fit_transform(development[representation])
        validation_matrix = vectorizer.transform(validation[representation])
        feature_seconds = time.perf_counter() - started
        for c_value in config["c_values"]:
            for class_weight in config["class_weights"]:
                model = LogisticRegression(
                    C=float(c_value),
                    class_weight=class_weight,
                    max_iter=int(config["max_iter"]),
                    solver="lbfgs",
                    random_state=int(config["random_seed"]),
                )
                started = time.perf_counter()
                model.fit(development_matrix, development["category"])
                fit_seconds = time.perf_counter() - started
                predictions = model.predict(validation_matrix)
                probabilities = model.predict_proba(validation_matrix)
                metrics = classification_metrics(
                    validation["category"].to_numpy(),
                    predictions,
                    probabilities,
                    model.classes_,
                )
                weight_name = "none" if class_weight is None else str(class_weight)
                key = (representation, float(c_value), weight_name)
                fitted[key] = (vectorizer, model)
                search_rows.append(
                    {
                        "representation": representation,
                        "c": float(c_value),
                        "class_weight": weight_name,
                        "features": int(development_matrix.shape[1]),
                        "feature_fit_seconds": feature_seconds,
                        "model_fit_seconds": fit_seconds,
                        "iterations": int(model.n_iter_.max()),
                        **metrics,
                    }
                )

    search = pd.DataFrame(search_rows).sort_values(
        ["macro_f1", "weighted_f1", "model_fit_seconds"], ascending=[False, False, True]
    )
    selected = search.iloc[0].to_dict()
    selected_key = (
        str(selected["representation"]),
        float(selected["c"]),
        str(selected["class_weight"]),
    )
    vectorizer, model = fitted[selected_key]
    pipeline = Pipeline([("features", vectorizer), ("classifier", model)])
    model_dir = private_root / "models" / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "tfidf_logistic.joblib"
    joblib.dump(pipeline, artifact_path, compress=3)

    private_results = private_root / "validation"
    private_results.mkdir(parents=True, exist_ok=True)
    search.to_csv(private_results / "baseline_search.csv", index=False)
    selected_model_metadata = {
        "model": "word-character TF-IDF multinomial logistic regression",
        "selection_metric": config["selection_metric"],
        "selected": selected,
        "majority_reference": _majority_metrics(development, validation),
        "development_rows": len(development),
        "validation_rows": len(validation),
        "official_test_used": False,
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "config_sha256": sha256_file(project_root / "configs" / "baseline.json"),
    }
    (private_results / "baseline_validation.json").write_text(
        json.dumps(selected_model_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    public_search = search.copy()
    public_search.to_csv(project_root / "reports" / "baseline_validation_search.csv", index=False)
    (project_root / "reports" / "baseline_validation.json").write_text(
        json.dumps(selected_model_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_manifest = {
        "relative_private_path": "models/baseline/tfidf_logistic.joblib",
        "sha256": selected_model_metadata["artifact_sha256"],
        "bytes": selected_model_metadata["artifact_bytes"],
        "stored_in_git": False,
    }
    (project_root / "manifests" / "baseline_artifact.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selected_model_metadata
