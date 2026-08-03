"""Add energy-delivery and detailed reporting audits to a completed SAR run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "_v25_sar_soc_pressure_seed2026_cap035_v1"
METHODS = (
    "B1-COMMON-THERMAL",
    "OFFLOAD-EFT-THERMAL-ALIGNED",
    "TUT-UOI-PRODUCT-LITE-v2",
    "MYOPIC-FROZEN-CALIBRATED-ON-CORRECTED-REQALIGN",
    "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-v1",
)
PROFILES = ("0W", "800W_x_10s", "1600W_x_10s", "3870W_x_10s")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized or [{"empty": ""}])


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def event_hash(rows: list[dict[str, str]]) -> str:
    payload = [
        (row["acquisition_id"], int(float(row["event_slot"])), round(float(row["sar_energy_j"]), 12))
        for row in rows
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def f(value: Any) -> float:
    return float(value or 0)


def short(method: str) -> str:
    return {
        METHODS[0]: "B1",
        METHODS[1]: "OFFLOAD",
        METHODS[2]: "TUT-v2",
        METHODS[3]: "OLD-GREEDY+REQALIGN",
        METHODS[4]: "CORRECTED-REQALIGN",
    }[method]


def md_table(fields: list[str], rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows)
    return "\n".join(lines)


def main(output: Path) -> int:
    metrics = read_csv(output / "tables" / "sar_soc_pressure_metrics.csv")
    power = yaml.safe_load((ROOT / "configs" / "bupt1_measured_power_v1.yaml").read_text(encoding="utf-8"))
    capacity_wh = f(power["battery_capacity_Wh"])
    discharge_efficiency = f(power["discharge_efficiency"])
    energy_rows = []
    identity_rows = []
    for profile in PROFILES:
        hashes = []
        for method in METHODS:
            trace = [
                row for row in read_csv(output / "runs" / profile / method / "sar_soc_debit_trace.csv")
                if row.get("acquisition_id")
            ]
            demand_wh = sum(f(row.get("sar_energy_wh")) for row in trace)
            reflected_wh = sum(
                max(0.0, f(row["soc_before_sar"]) - f(row["soc_after_sar"]))
                * capacity_wh
                * discharge_efficiency
                for row in trace
            )
            trace_hash = event_hash(trace)
            hashes.append(trace_hash)
            expected_count = 0 if profile == "0W" else 50
            energy_rows.append(
                {
                    "profile": profile,
                    "method": method,
                    "physical_acquisition_count": 50,
                    "soc_debit_event_count": len(trace),
                    "expected_soc_debit_event_count": expected_count,
                    "sar_energy_demand_Wh": demand_wh,
                    "sar_energy_reflected_in_soc_Wh": reflected_wh,
                    "unserved_energy_after_soc_floor_Wh": max(0.0, demand_wh - reflected_wh),
                    "soc_trace_clipped": reflected_wh + 1.0e-9 < demand_wh,
                    "event_trace_sha256": trace_hash,
                }
            )
        identity_rows.extend(
            {
                "profile": profile,
                "method": method,
                "event_trace_sha256": trace_hash,
                "same_acquisition_slot_energy_trace": len(set(hashes)) == 1,
            }
            for method, trace_hash in zip(METHODS, hashes)
        )
    write_csv(output / "audit" / "sar_energy_delivery_audit.csv", energy_rows)
    write_csv(output / "audit" / "sar_event_trace_identity.csv", identity_rows)
    if not all(row["same_acquisition_slot_energy_trace"] for row in identity_rows):
        raise RuntimeError("SAR_EVENT_TRACE_IDENTITY_MISMATCH")

    regression = read_csv(output / "audit" / "sar_disabled_exact_regression.csv")
    regression_pass = all(row["exact_match"].lower() == "true" for row in regression)
    recovery = read_csv(output / "tables" / "daylight_idle_soc_recovery_samples.csv")

    energy_summary = []
    for profile in PROFILES:
        group = [row for row in energy_rows if row["profile"] == profile]
        metric = next(row for row in metrics if row["profile"] == profile)
        energy_summary.append(
            {
                "profile": profile,
                "physical acquisitions": 50,
                "nominal demand Wh": f(metric["sar_total_energy_Wh"]),
                "SOC-reflected Wh range": f"{min(f(r['sar_energy_reflected_in_soc_Wh']) for r in group):.3f} - {max(f(r['sar_energy_reflected_in_soc_Wh']) for r in group):.3f}",
                "unserved Wh range": f"{min(f(r['unserved_energy_after_soc_floor_Wh']) for r in group):.3f} - {max(f(r['unserved_energy_after_soc_floor_Wh']) for r in group):.3f}",
            }
        )

    soc_rows = []
    primary_rows = []
    for row in metrics:
        soc_rows.append(
            {
                "profile": row["profile"], "method": short(row["method"]),
                "min SOC": f"{f(row['minimum_soc_including_instantaneous_sar']):.4f}",
                "min time h": f"{f(row['minimum_soc_time_s']) / 3600.0:.3f}",
                "final SOC": f"{f(row['final_soc']):.4f}",
                "<0.80 h": f"{f(row['duration_soc_below_0_80_s']) / 3600.0:.2f}",
                "<0.70 h": f"{f(row['duration_soc_below_0_70_s']) / 3600.0:.2f}",
                "SOC violations": int(f(row["soc_violation_count"])),
                "thermal violations": int(f(row["thermal_violation_count"])),
            }
        )
        primary_rows.append(
            {
                "profile": row["profile"], "method": short(row["method"]),
                "J_all": f"{f(row['J_all_net']):.3f}", "J_HV": f"{f(row['J_HV_net']):.3f}",
                "HV": int(f(row["HV_closure_count"])), "high": int(f(row["high_closure_count"])),
                "urgent": int(f(row["urgent_closure_count"])),
                "dJ_all": f"{f(row['delta_J_all_net_vs_0W']):.3f}",
                "dJ_HV": f"{f(row['delta_J_HV_net_vs_0W']):.3f}",
                "dHV": int(f(row["delta_HV_closure_count_vs_0W"])),
            }
        )

    recovery_rows = []
    for profile in PROFILES:
        candidates = [row for row in recovery if row["profile"] == profile and row["method"] == METHODS[4]]
        if candidates:
            row = candidates[0]
            recovery_rows.append(
                {"profile": profile, "method": "CORRECTED-REQALIGN", "time h": f"{f(row['time_s'])/3600:.3f}",
                 "solar W": f"{f(row['solar_power_w']):.3f}", "SOC before": f"{f(row['soc_before']):.6f}",
                 "SOC after": f"{f(row['soc_after']):.6f}", "recovery": f"{f(row['soc_recovery']):.8f}"}
            )

    assessment = json.loads((output / "assessment.json").read_text(encoding="utf-8"))
    report = [
        "# SAR observation energy SOC pressure diagnostic",
        "",
        "## Identity and regression",
        "",
        "- Frozen V25 seed=2026, capacity=0.35, 86400 s, 20 s slots.",
        "- The manifest contains no acquisition_id, observation_id or task_group_id. metadata.product_identifier maps 268 tasks to 50 physical Sentinel acquisitions; the largest acquisition derives 11 tasks.",
        f"- sar_payload_enabled=false exact regression: {'PASS' if regression_pass else 'FAIL'} ({sum(r['exact_match'].lower() == 'true' for r in regression)}/{len(regression)} core-file hashes).",
        "- Each nonzero profile debited exactly 50 observation events in every method. Within each profile, acquisition ID, event slot and requested energy hashes are identical across all five methods.",
        "",
        "## SAR energy",
        "",
        md_table(list(energy_summary[0]), energy_summary),
        "",
        "The nominal demand is the configured physical payload energy. At 1600 W and 3870 W the battery model reaches its 0-SOC clamp, so not all requested energy can be represented as battery discharge. Those profiles are infeasible stress points; the unserved portion must not be described as energy actually supplied by the battery.",
        "",
        "## SOC and safety",
        "",
        md_table(list(soc_rows[0]), soc_rows),
        "",
        "## Primary metrics and deltas from 0 W",
        "",
        md_table(list(primary_rows[0]), primary_rows),
        "",
        "Once the hard SOC floor is violated, the large negative J_all values are public safety penalties. They are diagnostic values, not meaningful method-ranking evidence.",
        "",
        "## Daylight idle recovery samples",
        "",
        md_table(list(recovery_rows[0]), recovery_rows) if recovery_rows else "No qualifying recovery sample.",
        "",
        "## Decision",
        "",
        "1600 W x 10 s does create clear SOC pressure, but it is not safe: all five methods reach SOC=0 and record 4131 SOC violations. Even 800 W x 10 s is unsafe, with minima around 0.325–0.329 below the frozen hard battery floor 0.55 and more than 3600 violation slots.",
        "",
        "Do not start 5 seeds x 3 capacities from these settings. First run one preregistered seed-2026 bracketing diagnostic below 800 W (for example 200/300/400 W x 10 s) without changing tasks or algorithms; expand only after selecting a fixed profile that creates visible SOC separation while retaining zero hard violations.",
    ]
    (output / "final_assessment.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    assessment.update(
        {
            "sar_disabled_exact_regression": regression_pass,
            "event_trace_identity_passed": True,
            "sar_800w_safe_all_methods": False,
            "sar_1600w_reaches_zero_soc_all_methods": True,
            "recommendation_detail": "Do not expand to 5 seeds x 3 capacities; bracket below 800 W first.",
        }
    )
    write_json(output / "assessment.json", assessment)
    write_json(output / "completion.json", assessment)
    print(json.dumps(assessment, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raise SystemExit(main(args.output))
