"""Failure-only diagnostics for the native-direct diagnostic replay.

The probe observes execution and persists evidence.  It never supplies a
fallback value and never catches an exception without re-raising it.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .atomic_io import atomic_write_json, atomic_write_text
except ImportError:
    from atomic_io import atomic_write_json, atomic_write_text


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    return repr(value)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _exception_tree(error: BaseException | None) -> Any:
    if error is None:
        return None
    return {
        "type": type(error).__name__,
        "message": str(error),
        "repr": repr(error),
        "cause": _exception_tree(error.__cause__),
        "context": _exception_tree(error.__context__),
        "suppress_context": bool(error.__suppress_context__),
    }


class DiagnosticProbe:
    def __init__(self, output: Path) -> None:
        self.output = Path(output)
        self.checkpoint: dict[str, Any] = {
            "current_slot": None,
            "current_simulation_time_s": None,
            "current_task_id": None,
            "current_queue_mb": None,
            "current_soc": None,
            "current_thermal": None,
            "current_contact": None,
            "last_successful_slot": None,
            "last_successful_simulation_time_s": None,
            "last_successful_task": None,
            "last_successful_event": None,
            "current_native_event_count": 0,
            "queue_mb": None,
            "soc": None,
            "thermal": None,
            "contact": None,
        }
        self.candidate_a_recorded = False
        self.candidate_b_recorded = False
        self.other_site_recorded = False
        self._previous_trace = None

    def note_step_started(self, timeline_step: Any, resource_state: Any, comm_state: Any) -> None:
        task = _get(timeline_step, "task", None)
        slot = int(_get(timeline_step, "slot_index", -1))
        seconds = float(_get(timeline_step, "slot_seconds", 0.0))
        self.checkpoint.update({
            "current_slot": slot,
            "current_simulation_time_s": slot * seconds,
            "current_task_id": _get(task, "task_id", None) if task is not None else None,
            "current_queue_mb": _get(comm_state, "queue_mb", None),
            "current_soc": _get(resource_state, "battery_soc", None),
            "current_thermal": _get(resource_state, "thermal_pressure", None),
            "current_contact": int(bool(_get(_get(timeline_step, "link_trace", None), "in_contact", False))),
        })

    def note_step(self, record: Any, slot_seconds: float | None = None) -> None:
        slot = int(_get(record, "slot", _get(record, "slot_index", -1)))
        task_id = str(_get(record, "task_id", "") or "")
        delivered_ids = str(_get(record, "delivered_task_ids", "") or "")
        delivered_products = str(_get(record, "delivered_product_types", "") or "")
        delivered_times = str(_get(record, "delivered_task_event_times_s", "") or "")
        if delivered_ids:
            count = len([item for item in delivered_ids.split(";") if item])
            self.checkpoint["current_native_event_count"] += count
            self.checkpoint["last_successful_event"] = {
                "task_ids": delivered_ids,
                "product_types": delivered_products,
                "event_times_s": delivered_times,
            }
        self.checkpoint.update({
            "last_successful_slot": slot,
            "last_successful_simulation_time_s": (
                _get(record, "time_s", None) if _get(record, "time_s", None) is not None
                else (slot * float(slot_seconds) if slot_seconds is not None else None)),
            "last_successful_task": task_id or self.checkpoint["last_successful_task"],
            "queue_mb": float(_get(record, "queue_next_mb", _get(record, "queue_mb", 0.0))),
            "soc": float(_get(record, "battery_next_soc", _get(record, "battery_soc", 0.0))),
            "thermal": float(_get(record, "thermal_next", _get(record, "Theta_t", 0.0))),
            "contact": int(float(_get(record, "C_t", 0.0)) > 0.0),
        })

    def note_native_event(self, index: int, task_id: Any, row: Mapping[str, Any]) -> None:
        self.checkpoint["current_native_event_count"] = max(
            int(self.checkpoint.get("current_native_event_count") or 0), index + 1)
        self.checkpoint["last_successful_event"] = {
            "native_event_index": index, "task_id": _safe(task_id), "event": _safe(dict(row))}

    def _runtime_from_stack(self, frame: Any) -> dict[str, Any]:
        current = frame
        while current is not None:
            local = current.f_locals
            step = local.get("timeline_step")
            if step is not None:
                task = _get(step, "task", None)
                return {
                    "slot": _get(step, "slot_index", None),
                    "simulation_time_s": (
                        float(_get(step, "slot_index", 0)) * float(_get(step, "slot_seconds", 0.0))),
                    "task_id": _get(task, "task_id", None) if task is not None else None,
                    "pre_filter_count": len(local.get("actions", []) or []),
                    "safe_filter_count": len(local.get("safe_actions", []) or []),
                    "post_filter_count": len(local.get("candidate_actions", []) or []),
                }
            current = current.f_back
        return {}

    @contextlib.contextmanager
    def patched_step_runner(self, step_runner_type: Any):
        original = step_runner_type.run_step
        probe = self

        def diagnostic_run_step(instance, timeline_step, *args, **kwargs):
            resource_state = args[0] if len(args) > 0 else kwargs.get("resource_state")
            comm_state = args[2] if len(args) > 2 else kwargs.get("comm_state")
            probe.note_step_started(timeline_step, resource_state, comm_state)
            result = original(instance, timeline_step, *args, **kwargs)
            probe.note_step(result[0], _get(timeline_step, "slot_seconds", None))
            return result

        step_runner_type.run_step = diagnostic_run_step
        try:
            yield
        finally:
            step_runner_type.run_step = original

    @contextlib.contextmanager
    def patched_router(self, router_type: Any):
        original = router_type.select_action
        probe = self

        def diagnostic_select_action(instance, *args, **kwargs):
            previous = sys.gettrace()
            sys.settrace(probe._trace)
            try:
                return original(instance, *args, **kwargs)
            finally:
                sys.settrace(previous)

        router_type.select_action = diagnostic_select_action
        try:
            yield
        finally:
            router_type.select_action = original

    def record_candidate_a(self, rows: Iterable[Mapping[str, Any]], selected_name: Any,
                           runtime: Mapping[str, Any] | None = None,
                           scoring_mode: Any = None) -> None:
        if self.candidate_a_recorded:
            return
        rows = list(rows)
        scores = [row.get("final_action_score") for row in rows]
        payload = {
            "candidate": "A",
            "site": "SafeLinUCBRouter.select_action",
            "slot": (runtime or {}).get("slot"),
            "simulation_time_s": (runtime or {}).get("simulation_time_s"),
            "task_id": (runtime or {}).get("task_id"),
            "row_count": len(rows),
            "selected_name": _safe(selected_name),
            "action_names": [_safe(row.get("action_name")) for row in rows],
            "final_action_scores": _safe(scores),
            "has_nan": any(isinstance(value, float) and math.isnan(value) for value in scores),
            "has_inf": any(isinstance(value, float) and math.isinf(value) for value in scores),
            "empty_candidate_set": len(rows) == 0,
            "pre_filter_count": (runtime or {}).get("pre_filter_count"),
            "safe_filter_count": (runtime or {}).get("safe_filter_count"),
            "post_filter_count": (runtime or {}).get("post_filter_count"),
            "scoring_mode": _safe(scoring_mode if scoring_mode is not None else (runtime or {}).get("scoring_mode")),
            "score_components": [
                {
                    "action_name": _safe(row.get("action_name")),
                    "utility_prior_raw": _safe(row.get("utility_prior_raw")),
                    "utility_prior_norm": _safe(row.get("utility_prior_norm")),
                    "learned_correction_raw": _safe(row.get("learned_correction_raw")),
                    "learned_correction_norm": _safe(row.get("learned_correction_norm")),
                    "safe_exploration_raw": _safe(row.get("safe_exploration_raw")),
                    "safe_exploration_norm": _safe(row.get("safe_exploration_norm")),
                    "final_action_score": _safe(row.get("final_action_score")),
                } for row in rows
            ],
        }
        atomic_write_json(self.output / "candidate_a_diagnostic.json", payload)
        self.candidate_a_recorded = True

    def record_candidate_b(self, tasks: Iterable[Mapping[str, Any]], task_id: Any,
                           event_index: int, row: Mapping[str, Any]) -> None:
        if self.candidate_b_recorded:
            return
        tasks = tuple(tasks)
        ids = [str(item.get("task_id")) for item in tasks]
        canonical_hash = hashlib.sha256(
            json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        raw = str(task_id)
        normalized = raw.strip()
        payload = {
            "candidate": "B",
            "site": "normalize_episode canonical task lookup",
            "native_event_index": int(event_index),
            "task_id": _safe(task_id),
            "task_id_repr": repr(task_id),
            "task_id_type": type(task_id).__name__,
            "task_id_length": len(raw),
            "raw_task_id": raw,
            "normalized_task_id": normalized,
            "canonical_task_count": len(tasks),
            "canonical_task_id_sequence_sha256": canonical_hash,
            "whitespace_difference": raw != normalized,
            "case_insensitive_matches": [item for item in ids if item.casefold() == normalized.casefold()],
            "prefix_matches": [item for item in ids if item.startswith(normalized) or normalized.startswith(item)][:20],
            "suffix_matches": [item for item in ids if item.endswith(normalized) or normalized.endswith(item)][:20],
            "exact_type_difference": any(item == raw and type(task_id) is not str for item in ids),
            "native_event": _safe(dict(row)),
        }
        atomic_write_json(self.output / "candidate_b_diagnostic.json", payload)
        self.candidate_b_recorded = True

    def _trace(self, frame: Any, event: str, arg: Any):
        raw_filename = str(frame.f_code.co_filename)
        filename = (raw_filename if raw_filename.startswith("<")
                    else str(Path(raw_filename).resolve())).replace("\\", "/").lower()
        function = frame.f_code.co_name
        is_router = function == "select_action" and filename.endswith("/src/routers/safe_linucb_router.py")
        is_normalize = function == "normalize_episode" and filename.endswith("/native_support.py")
        if event == "call":
            return self._trace if (is_router or is_normalize) else None
        if event == "exception" and isinstance(arg, tuple) and len(arg) >= 2 and arg[0] is StopIteration:
            if is_router:
                runtime = self._runtime_from_stack(frame)
                scoring = "utility_guided" if frame.f_locals.get("use_utility_guided") else "legacy"
                self.record_candidate_a(frame.f_locals.get("rows", []),
                                        frame.f_locals.get("selected_name"), runtime, scoring)
            elif not (is_router or is_normalize):
                if not self.other_site_recorded:
                    atomic_write_json(self.output / "other_stopiteration_site.json", {
                        "candidate": "OTHER", "file": str(Path(frame.f_code.co_filename).resolve()),
                        "function": frame.f_code.co_name, "line": frame.f_lineno,
                        "runtime": _safe(self._runtime_from_stack(frame)),
                    })
                    self.other_site_recorded = True
        return self._trace

    @contextlib.contextmanager
    def installed(self):
        self._previous_trace = sys.gettrace()
        sys.settrace(self._trace)
        try:
            yield self
        finally:
            sys.settrace(self._previous_trace)


def diagnostic_next_action(rows: Iterable[Mapping[str, Any]], selected_name: Any, *,
                           probe: DiagnosticProbe, runtime: Mapping[str, Any] | None = None):
    try:
        return next(row for row in rows if row["action_name"] == selected_name)
    except StopIteration:
        probe.record_candidate_a(rows, selected_name, runtime)
        raise


def diagnostic_canonical_task(tasks: Iterable[Mapping[str, Any]], task_id: Any,
                              event_index: int, row: Mapping[str, Any], probe: DiagnosticProbe):
    try:
        return next(item for item in tasks if item["task_id"] == task_id)
    except StopIteration:
        probe.record_candidate_b(tasks, task_id, event_index, row)
        raise


def persist_failure_bundle(output: Path, error: BaseException, probe: DiagnosticProbe,
                           stdout_text: str = "", stderr_text: str = "") -> None:
    output = Path(output)
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    atomic_write_text(output / "full_traceback.txt", trace)
    atomic_write_text(output / "stdout.txt", stdout_text)
    atomic_write_text(output / "stderr.txt", stderr_text + trace)
    atomic_write_json(output / "exception_diagnostic.json", _exception_tree(error))
    atomic_write_json(output / "last_runtime_checkpoint.json", _safe(probe.checkpoint))
