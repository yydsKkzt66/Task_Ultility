"""Verify the competition headline results from frozen simulation tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE = (
    ROOT
    / "outputs"
    / "_current_prop_four_baselines_fiveseed_dual_thermal_v1"
    / "attempt_001"
    / "tables"
    / "mean_std_metrics.csv"
)
PROP = "PROP-V2-SAFE-METADATA-CURRENT-v1"
TUT = "TUT-PROXY-THERMAL-CORRECTED"

EXPECTED = {
    "J_all_net_mean": 44.908707891565584,
    "J_HV_net_mean": 37.92065432236117,
    "HV_closure_count_mean": 56.2,
    "controllable_energy_Wh_mean": 2.3385667419490335,
    "transmitted_MB_mean": 167.1895956432671,
}


def main() -> int:
    with TABLE.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {row["method"]: row for row in csv.DictReader(stream)}

    missing = sorted({PROP, TUT} - set(rows))
    if missing:
        print(json.dumps({"status": "FAIL", "missing_methods": missing}, indent=2))
        return 2

    prop = rows[PROP]
    tut = rows[TUT]
    mismatches = {}
    if int(prop["seed_count"]) != 5 or int(tut["seed_count"]) != 5:
        mismatches["seed_count"] = {
            "expected": 5,
            "prop": int(prop["seed_count"]),
            "tut": int(tut["seed_count"]),
        }
    for field, expected in EXPECTED.items():
        actual = float(prop[field])
        if abs(actual - expected) > 1e-12:
            mismatches[field] = {"expected": expected, "actual": actual}

    comparisons = {
        "overall_utility_gain_vs_tut_percent":
            (float(prop["J_all_net_mean"]) / float(tut["J_all_net_mean"]) - 1.0) * 100.0,
        "high_value_utility_gain_vs_tut_percent":
            (float(prop["J_HV_net_mean"]) / float(tut["J_HV_net_mean"]) - 1.0) * 100.0,
        "energy_reduction_vs_tut_percent":
            (1.0 - float(prop["controllable_energy_Wh_mean"]) / float(tut["controllable_energy_Wh_mean"])) * 100.0,
        "downlink_reduction_vs_tut_percent":
            (1.0 - float(prop["transmitted_MB_mean"]) / float(tut["transmitted_MB_mean"])) * 100.0,
    }
    status = "PASS" if not mismatches else "FAIL"
    print(json.dumps({
        "status": status,
        "source_table": str(TABLE.relative_to(ROOT)),
        "seed_count": int(prop["seed_count"]),
        "prop": {field: float(prop[field]) for field in EXPECTED},
        "comparisons": comparisons,
        "mismatches": mismatches,
    }, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
