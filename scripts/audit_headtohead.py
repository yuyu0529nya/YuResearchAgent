#!/usr/bin/env python3
"""Verify a head-to-head artifact without making API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.preregistration import audit_headtohead_artifact, load_preregistration


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a preregistered result artifact")
    parser.add_argument("artifact")
    parser.add_argument(
        "--preregistration",
        default="docs/evaluation/headtohead_v4_preregistration.json",
    )
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    preregistration = load_preregistration(args.preregistration)
    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    result = audit_headtohead_artifact(
        artifact,
        preregistration,
        project_root=PROJECT_ROOT,
        require_complete=args.require_complete,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
