"""
CLI entrypoint for the Pramāṇa-Constrained Inference Engine.

Usage
-----
    python -m pramana_engine.cli --input input.json
    echo '<json>' | python -m pramana_engine.cli

The JSON input must have the shape::

    {
        "proposition": { <Proposition fields> },
        "evidence": [ { <Evidence fields> }, ... ],
        "hetu": "smoke",       // optional
        "sadhya": "fire",      // optional (required for Anumāna)
        "example_subject": "kitchen"  // optional
    }

Output is printed to stdout as JSON with three top-level keys:

    - ``"verdict"``        – verdict label string
    - ``"normalized"``     – normalised proposition/evidence summary
    - ``"reasoning_trace"``– full NetworkX DAG in node-link JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from pramana_engine.engine import PramanaEngine


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pramana-engine",
        description="Pramāṇa-Constrained Inference Engine CLI",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="Path to JSON input file (defaults to stdin).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Epistemic confidence threshold (default: 0.5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI main entry point."""
    args = _parse_args(argv)

    try:
        raw: Dict[str, Any] = json.load(args.input)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON input – {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if args.input is not sys.stdin:
            args.input.close()

    proposition_data = raw.get("proposition")
    if proposition_data is None:
        print("ERROR: Input JSON must contain a 'proposition' key.", file=sys.stderr)
        sys.exit(1)

    engine = PramanaEngine(confidence_threshold=args.threshold)
    output = engine.run(
        proposition_data=proposition_data,
        evidence_data=raw.get("evidence", []),
        subject=raw.get("subject"),
        hetu=raw.get("hetu"),
        sadhya=raw.get("sadhya"),
        example_subject=raw.get("example_subject"),
    )

    result: Dict[str, Any] = {
        "verdict": output.verdict.label.value,
        "elapsed_time_ms": round(output.elapsed_time_ms, 3),
        "normalized": output.to_dict(),
        "reasoning_trace": json.loads(output.trace_json),
    }

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
