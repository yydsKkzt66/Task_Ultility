from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.run_fixed_product_compression_probe_seed2026 import REQUIRED_FILES, generate_probe


def test_probe_entrypoint_can_import_src_when_loaded_by_file_path() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_fixed_product_compression_probe_seed2026.py"
    code = f"ns=__import__('runpy').run_path({str(script)!r}); ns['_run_sentinel']()"
    completed = subprocess.run([sys.executable, "-c", code], cwd=script.parent, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_probe_emits_required_blocked_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "probe"
    generate_probe(output_dir=output, repository_root=Path(__file__).resolve().parents[1])

    assert set(REQUIRED_FILES) <= {path.name for path in output.iterdir() if path.is_file()}
    assert "BLOCKED_NOT_RUN" in (output / "fixed_profile_selection.md").read_text(encoding="utf-8")
    assert "BLOCKED_PROPAGATION_FAILURE" in (output / "active_trial_method_summary.csv").read_text(encoding="utf-8")
    capability = (output / "codec_capability_probe.md").read_text(encoding="utf-8")
    assert "OpenJPEG" in capability
    assert "HTJ2K" in capability
    assert "not available" in capability
    contract = (output / "future_real_product_integration_contract.md").read_text(encoding="utf-8")
    assert "crop geometry" in contract
    assert "decoded product quality" in contract
    assert (output / "rollback" / "file_hashes_after.csv").is_file()
    for name in (
        "encoded_propagation_usage_inventory.csv",
        "encoded_payload_lineage_trace.csv",
        "encoded_physics_recalculation_audit.csv",
    ):
        assert (output / name).is_file()
    lineage = (output / "encoded_payload_lineage_trace.csv").read_text(encoding="utf-8")
    assert "encoding_completed" in lineage
    assert "partially_transmitted" in lineage
    assert "delivered" in lineage
    recalculation = (output / "encoded_physics_recalculation_audit.csv").read_text(encoding="utf-8")
    assert "MATCH" in recalculation
    final_report = (output / "final_compression_probe_report.md").read_text(encoding="utf-8")
    assert "Propagation matrix" in final_report
    assert "BLOCKED_PROPAGATION_FAILURE" in final_report
