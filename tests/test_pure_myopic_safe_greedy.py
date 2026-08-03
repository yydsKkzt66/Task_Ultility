from types import SimpleNamespace

from src.routers.pure_myopic_safe_greedy import (
    install_pure_myopic_safe_greedy,
    select_pure_myopic,
)


def action(name, model="balanced", product="roi"):
    return SimpleNamespace(action_name=name, model_name=model, payload_policy=product)


def predicted(score, latency=1.0, payload=1.0, energy=1.0):
    return SimpleNamespace(
        predicted_reward=score, predicted_L_inf=latency, predicted_L_queue=0.0,
        predicted_L_comm=0.0, predicted_S_gen_mb=payload,
        predicted_E_comp=energy, predicted_E_comm=0.0,
    )


def test_best_current_net_wins():
    actions = [action("a"), action("b")]
    selected, _ = select_pure_myopic(actions, {"a": predicted(.2), "b": predicted(.3)})
    assert selected.action_name == "b"


def test_defer_zero_blocks_negative_work():
    actions = [action("work"), action("delay/store", "delay_store", "defer")]
    selected, _ = select_pure_myopic(actions, {"work": predicted(-.1), "delay/store": predicted(0, 0, 0, 0)})
    assert selected.action_name == "delay/store"


def test_equal_net_prefers_resource_lean_action():
    actions = [action("large"), action("small")]
    selected, _ = select_pure_myopic(actions, {"large": predicted(.5, payload=2), "small": predicted(.5, payload=1)})
    assert selected.action_name == "small"


def test_installer_audits_nonzero_guidance_and_blocks_learning():
    delegate, audit = SimpleNamespace(), {}
    install_pure_myopic_safe_greedy(delegate, audit)
    actions = [action("a"), action("delay/store", "delay_store", "defer")]
    selected = delegate.select_action(None, actions, {"a": predicted(.1), "delay/store": predicted(0, 0, 0, 0)}, {"lambda_E": 1.0})
    delegate.update(None, selected, 1.0)
    assert audit["nonzero_guidance_violation_count"] == 1
    assert audit["blocked_online_update_count"] == 1
