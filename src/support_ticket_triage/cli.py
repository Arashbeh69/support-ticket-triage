"""Command-line prediction and metadata interface."""

from __future__ import annotations

import argparse
import json

from support_ticket_triage.inference import InferenceEngine, InputValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local banking-support intent triage")
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict", help="predict one short message")
    predict.add_argument("--text", required=True, help="message scored locally; never transmitted")
    subparsers.add_parser("metadata", help="show the frozen local model contract")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    engine = InferenceEngine.from_environment()
    if args.command == "metadata":
        print(json.dumps(engine.metadata(), indent=2, sort_keys=True))
        return 0
    try:
        result = engine.predict(args.text)
    except InputValidationError as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 2
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
