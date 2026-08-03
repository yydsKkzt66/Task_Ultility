# Long-Term Design Spec

## Scope

Project theme:

Task-Utility-Driven Computation-Communication Co-Design for Resource-Constrained LEO Onboard Intelligence

This project is not a generic LEO MEC offloading system, not a generic resource allocation benchmark, and not a single-objective optimizer for energy, latency, or throughput alone.

The core objective is to jointly decide:

`a_t = (m_t, p_t)`

where:

- `m_t`: onboard model profile / model complexity
- `p_t`: payload granularity / delivery option

The end objective is to maximize effective task delivery utility: whether high-value tasks are reliably inferred, generate sufficient information, and are delivered before deadline under limited contact windows.

## Mandatory Five-Module Logic

The project must preserve this end-to-end logic:

1. Task & Context Construction
2. Hierarchical Task-Utility Joint Decision
3. Execution & Payload Formation
4. Contact-Aware Delivery
5. Closed-Loop Update

## Long-Term Engineering Rules

1. Do not rewrite the project from scratch. Extend the existing runnable demo incrementally.
2. Keep the joint action definition `a_t = (m_t, p_t)`. The policy cannot choose only model or only payload.
3. Safety Filter must run before Utility Selection / Bandit Selection.
4. Do not allow unsafe baselines to obtain fake high utility through repeated safety violations.
5. In the current WC case study, model `E/L/Q` can use base profiles:
   `m -> {Q_base, E_base, L_base}`
   Lightweight corrections are allowed, such as task difficulty affecting `Q` and thermal state affecting `L`.
6. Battery, queue, and communication window states should mainly affect resource price, safety feasibility, and delivery value.
   They should not be modeled as tiny battery percentage changes directly causing large changes in physical model energy.
7. The default proposed method does not require DDQN training in the current WC setting.
8. The default proposed method must be:
   `proposed_lyapunov_rule_mode_safe_bandit`
9. DDQN is only an optional learning-based mode controller.
   It must replace `RuleBasedModeController` through the same `BaseModeController` interface.
10. Do not describe the default method as DDQN-based unless `DDQNModeController` is actually trained, tested, and enabled in config.
11. Lyapunov-inspired base pricing is formula-driven / state-driven and does not require training.
12. Safe Bandit / LinUCB does not require offline training, but may update online.
13. If DDQN is enabled, it may only learn:
   `s_k_long -> c_k`
   DDQN is not allowed to output task-level actions `a_t = (m_t, p_t)` directly.
14. Keep parameters centralized in config as much as possible for manual tuning and sensitivity analysis.
15. After each Phase, the following must run:
   `python -m pytest -q`
   `python -m src.run_demo`
16. After each Phase, report:
   - modified files
   - why they were changed
   - how to run
   - remaining TODOs

## Phase Workflow Constraint

- This document is a long-term design constraint, not a request to implement everything at once.
- Each future request should only implement the specific Phase given by the user.
- Each Phase must remain runnable after completion.

## Default Interpretation Rule

Unless a later Phase explicitly overrides a narrow design choice, all future work should stay consistent with this document.
