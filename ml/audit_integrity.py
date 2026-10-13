#!/usr/bin/env python3
"""Finalize dataset integrity: fix cross-split exact dups, scan near candidates, audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.config import load_config
from dataset.integrity import run_integrity_audit
from dataset.repair_splits import regenerate_group_aware_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FreshNet dataset integrity finalization")
    parser.add_argument("--config", default=str(ROOT.parent / "configs" / "dataset.yaml"))
    parser.add_argument("--skip-near-scan", action="store_true")
    parser.add_argument(
        "--repair-splits",
        action="store_true",
        help="Regenerate manifests so exact/heuristic groups never cross splits.",
    )
    parser.add_argument(
        "--force-repair",
        action="store_true",
        help="Repair splits even if the pre-check finds no exact leaks.",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    manifests_dir = config.manifests_dir

    pre = run_integrity_audit(
        manifests_dir=manifests_dir,
        config=config,
        run_near_scan=False,
        write_report=False,
    )
    needs_repair = pre["exact_duplicate_cross_split_groups"] > 0 or pre["split_group_cross_split_groups"] > 0
    repaired = False
    if args.repair_splits or args.force_repair or needs_repair:
        repair = regenerate_group_aware_splits(manifests_dir, ratios=config.split_ratios, seed=config.split_seed)
        repaired = True
        print(json.dumps({"repair": repair}, indent=2))
    else:
        print(
            json.dumps(
                {
                    "repair": None,
                    "note": "Existing splits preserved (no exact/heuristic cross-split groups detected).",
                },
                indent=2,
            )
        )

    final = run_integrity_audit(
        manifests_dir=manifests_dir,
        config=config,
        run_near_scan=not args.skip_near_scan,
    )
    final["splits_repaired"] = repaired
    print(json.dumps(
        {
            "report_path": final["report_path"],
            "exact_duplicate_cross_split_groups": final["exact_duplicate_cross_split_groups"],
            "split_group_cross_split_groups": final["split_group_cross_split_groups"],
            "near_complete": final["integrity_status"]["near_duplicate_scan_complete"],
            "provisional_experiments_only": final["integrity_status"]["provisional_experiments_only"],
            "counts": final["counts"],
            "split_ratio_audit": final["split_ratio_audit"],
        },
        indent=2,
    ))
    # Non-zero if exact/heuristic leakage remains.
    if final["exact_duplicate_cross_split_groups"] or final["split_group_cross_split_groups"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
