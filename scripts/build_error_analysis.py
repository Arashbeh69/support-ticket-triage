"""Create aggregate, text-free error analysis and a private review queue."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from support_ticket_triage.text import template_key

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.with_name(f"{ROOT.name}-private")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _routing_lookup() -> dict[str, str]:
    groups = _json(ROOT / "configs" / "routing_groups.json")
    return {label: group for group, labels in groups.items() for label in labels}


def _support_bin(value: int) -> str:
    if value < 75:
        return "development support <75"
    if value < 125:
        return "development support 75-124"
    return "development support >=125"


def _length_bin(value: int) -> str:
    if value <= 5:
        return "1-5 words"
    if value <= 10:
        return "6-10 words"
    if value <= 20:
        return "11-20 words"
    return "21+ words"


def _aggregate(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupby(column, dropna=False)
        .agg(rows=("error", "size"), errors=("error", "sum"))
        .assign(error_rate=lambda data: data["errors"] / data["rows"])
        .reset_index()
    )


def main() -> None:
    results = _json(ROOT / "reports" / "official_test_results.json")
    evaluation = _json(ROOT / "configs" / "evaluation.json")
    champion = evaluation["champion"]
    predictions = pd.read_csv(PRIVATE / "validation" / "official_test_predictions.csv")
    model_rows = pd.read_csv(PRIVATE / "data" / "modelling" / "model_rows.csv")
    audit_rows = pd.read_csv(PRIVATE / "data" / "normalized" / "audit_rows.csv")
    test = (
        model_rows.loc[model_rows["source_split"] == "test"]
        .drop(columns=["privacy_flag"])
        .merge(
            audit_rows[["row_id", "text", "normalized_text", "audit_key"]],
            on="row_id",
            how="inner",
            validate="one_to_one",
        )
    )
    frame = predictions.merge(test, on="row_id", how="inner", validate="one_to_one")
    frame["error"] = frame["champion_prediction"] != frame["true_label"]
    routing = _routing_lookup()
    frame["true_routing_group"] = frame["true_label"].map(routing)
    frame["predicted_routing_group"] = frame["champion_prediction"].map(routing)
    frame["routing_error"] = frame["true_routing_group"] != frame["predicted_routing_group"]

    train_support = pd.read_csv(ROOT / "reports" / "class_distribution.csv")
    train_support = (
        train_support.loc[train_support["source_split"] == "train"]
        .set_index("category")["count"]
        .to_dict()
    )
    frame["development_support"] = frame["true_label"].map(train_support)
    frame["support_bin"] = frame["development_support"].map(_support_bin)
    frame["word_count"] = frame["redacted_text"].fillna("").str.split().str.len()
    frame["length_bin"] = frame["word_count"].map(_length_bin)
    frame["confidence_bin"] = pd.cut(
        frame["champion_confidence"],
        bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
        right=True,
    ).astype(str)

    all_audit = audit_rows.copy()
    all_audit["template_key"] = all_audit["text"].map(template_key)
    exact_counts = all_audit["audit_key"].value_counts()
    template_stats = all_audit.groupby("template_key").agg(
        rows=("row_id", "size"), distinct_exact=("audit_key", "nunique")
    )
    frame["template_key"] = frame["text"].map(template_key)
    frame["duplicate_status"] = np.where(
        frame["audit_key"].map(exact_counts).fillna(0) > 1,
        "exact duplicate/template",
        np.where(
            frame["template_key"].map(template_stats["rows"]).fillna(0).gt(1)
            & frame["template_key"].map(template_stats["distinct_exact"]).fillna(0).gt(1),
            "generalized template",
            "unique under audit keys",
        ),
    )

    baseline = joblib.load(PRIVATE / "models" / "baseline" / "tfidf_logistic.joblib")
    word_vectorizer = dict(baseline.named_steps["features"].transformer_list)["word"]
    vocabulary = set(word_vectorizer.vocabulary_)

    def oov_ratio(value: str) -> float:
        tokens = re.findall(r"(?u)\b\w\w+\b", value.casefold())
        if not tokens:
            return 1.0
        return sum(token not in vocabulary for token in tokens) / len(tokens)

    frame["word_oov_ratio"] = frame["redacted_text"].fillna("").map(oov_ratio)
    frame["spelling_variation_proxy"] = frame["word_oov_ratio"] >= 0.5
    frame["missing_context_review_queue"] = frame["word_count"] <= 3
    frame["genuine_ambiguity_review_queue"] = frame["champion_top_two_margin"] < 0.10
    frame["possible_annotation_error_review_queue"] = (
        frame["error"]
        & (frame["champion_confidence"] >= 0.80)
        & (frame["duplicate_status"] != "unique under audit keys")
    )
    high_risk = set(evaluation["review_policy"]["high_risk_intents"])
    frame["high_risk_consequence"] = frame["true_label"].isin(high_risk) | frame[
        "champion_prediction"
    ].isin(high_risk)
    frame["redaction_impact_group"] = np.where(frame["privacy_flag"], "redacted", "not redacted")

    reports = ROOT / "reports"
    for column, filename in (
        ("true_routing_group", "errors_by_operational_group.csv"),
        ("support_bin", "errors_by_class_support.csv"),
        ("length_bin", "errors_by_text_length.csv"),
        ("confidence_bin", "errors_by_confidence_bin.csv"),
        ("duplicate_status", "errors_by_duplicate_status.csv"),
        ("spelling_variation_proxy", "errors_by_spelling_proxy.csv"),
        ("redaction_impact_group", "errors_by_redaction_status.csv"),
        ("high_risk_consequence", "errors_by_high_risk_status.csv"),
    ):
        _aggregate(frame, column).to_csv(reports / filename, index=False)

    confusion_pairs = (
        frame.loc[frame["error"]]
        .groupby(["true_label", "champion_prediction"])
        .size()
        .rename("errors")
        .reset_index()
        .sort_values(["errors", "true_label", "champion_prediction"], ascending=[False, True, True])
    )
    confusion_pairs.to_csv(reports / "major_confusion_pairs.csv", index=False)
    (
        frame.groupby(["true_routing_group", "predicted_routing_group"])
        .size()
        .rename("rows")
        .reset_index()
        .to_csv(reports / "routing_group_confusion.csv", index=False)
    )

    queue_columns = [
        "row_id",
        "text",
        "true_label",
        "champion_prediction",
        "champion_confidence",
        "champion_top_two_margin",
        "duplicate_status",
        "spelling_variation_proxy",
        "missing_context_review_queue",
        "genuine_ambiguity_review_queue",
        "possible_annotation_error_review_queue",
        "privacy_flag",
        "high_risk_consequence",
    ]
    frame.loc[frame["error"], queue_columns].to_csv(
        PRIVATE / "validation" / "official_test_error_review.csv", index=False
    )

    wrong = frame.loc[frame["error"]]
    summary = {
        "champion": champion,
        "test_rows": len(frame),
        "intent_errors": int(frame["error"].sum()),
        "routing_group_errors": int(frame["routing_error"].sum()),
        "top_confusion_pairs": confusion_pairs.head(15).to_dict(orient="records"),
        "review_queue_flags_among_errors": {
            "spelling_variation_proxy": int(wrong["spelling_variation_proxy"].sum()),
            "missing_context_proxy": int(wrong["missing_context_review_queue"].sum()),
            "genuine_ambiguity_proxy": int(wrong["genuine_ambiguity_review_queue"].sum()),
            "possible_annotation_error_proxy": int(
                wrong["possible_annotation_error_review_queue"].sum()
            ),
            "redacted": int(wrong["privacy_flag"].sum()),
            "high_risk_consequence": int(wrong["high_risk_consequence"].sum()),
        },
        "proxy_warning": (
            "Spelling, missing-context, ambiguity, and possible-annotation-error counts are "
            "conservative review-queue heuristics, not adjudicated labels. "
            "No official label was changed."
        ),
        "official_labels_changed": False,
        "selective_classification": results["selective_classification"],
    }
    (reports / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "ERROR_ANALYSIS.md").write_text(
        "# Structured error analysis\n\n"
        f"The frozen champion makes {summary['intent_errors']:,} intent errors across "
        f"{len(frame):,} official test rows and {summary['routing_group_errors']:,} errors "
        "that cross a project-defined routing-group boundary.\n\n"
        "## Dimensions\n\n"
        "Aggregate CSVs break error rate down by operational group, development support, "
        "text length, calibrated-confidence bin, duplicate/template status, a word-OOV "
        "spelling proxy, redaction status, and high-risk consequence. Raw text and row-level "
        "predictions remain private.\n\n"
        "## Semantic review queues\n\n"
        f"Among errors, {summary['review_queue_flags_among_errors']['missing_context_proxy']:,} "
        "have at most three words, "
        f"{summary['review_queue_flags_among_errors']['genuine_ambiguity_proxy']:,} have a "
        "top-two margin below 0.10, and "
        f"{summary['review_queue_flags_among_errors']['possible_annotation_error_proxy']:,} "
        "are high-confidence duplicate/template cases queued for possible annotation review. "
        "These are review heuristics, not adjudications. No official test label was changed.\n\n"
        "## Operational consequence\n\n"
        f"{summary['review_queue_flags_among_errors']['high_risk_consequence']:,} errors involve "
        "a configured high-risk true or predicted intent. Those predictions are always sent "
        "to human review by the portfolio policy. Confidence and abstention do not establish "
        "safe open-set behavior.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
