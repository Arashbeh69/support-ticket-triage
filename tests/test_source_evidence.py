import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_verified_source_manifest_matches_contract() -> None:
    contract = json.loads((ROOT / "configs" / "source.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "manifests" / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == contract["commit"]
    assert manifest["split_counts"] == {"test.csv": 3080, "train.csv": 10003}
    assert manifest["label_count"] == 77
    assert manifest["licence"] == "CC BY 4.0"
    assert manifest["files"]["train.csv"]["sha256"] == (
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
    )
    assert manifest["files"]["test.csv"]["sha256"] == (
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
    )
