"""Report-only lifecycle taxonomy for delivered onboard products."""

from __future__ import annotations

from dataclasses import dataclass


STAGES = ("alert", "confirmation", "evidence")


@dataclass(frozen=True)
class StageCredit:
    """Stage indicators granted by a delivered product under the common taxonomy."""

    alert: int
    confirmation: int
    evidence: int


def normalize_product_type(product_type: object) -> str:
    """Normalize simulator product labels without changing their semantics."""

    label = str(product_type or "").strip().lower().replace("-", "_")
    aliases = {
        "roi+ctx": "roi_context",
        "roi_ctx": "roi_context",
        "context": "roi_context",
        "metadata_only": "metadata",
        "defer": "defer",
        "delay_store": "defer",
    }
    return aliases.get(label, label)


def stage_credit_for_product(product_type: object) -> StageCredit:
    """Return fair, policy-agnostic lifecycle stage credit for one delivered product.

    Fairness rules:
    - metadata can satisfy alert only;
    - ROI can satisfy alert and confirmation;
    - ROI+Ctx can satisfy alert, confirmation, and evidence;
    - residual/context evidence products can satisfy evidence, and also close
      earlier stages if no earlier product arrived in time.
    """

    product = normalize_product_type(product_type)
    if product == "metadata":
        return StageCredit(alert=1, confirmation=0, evidence=0)
    if product == "roi":
        return StageCredit(alert=1, confirmation=1, evidence=0)
    if product == "roi_context":
        return StageCredit(alert=1, confirmation=1, evidence=1)
    if product in {"residual", "residual_roi_context", "context_evidence", "evidence"}:
        return StageCredit(alert=1, confirmation=1, evidence=1)
    if product in {"residual_roi", "verification"}:
        return StageCredit(alert=1, confirmation=1, evidence=0)
    return StageCredit(alert=0, confirmation=0, evidence=0)


def is_metadata_only(product_type: object) -> bool:
    return normalize_product_type(product_type) == "metadata"


def is_direct_roi_or_context(product_type: object) -> bool:
    return normalize_product_type(product_type) in {"roi", "roi_context"}
