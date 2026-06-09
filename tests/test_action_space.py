from src.actions.action_space import ActionSpace


def test_action_space_contains_five_joint_actions():
    actions = ActionSpace().list_actions()
    assert len(actions) == 5
    assert {action.action_name for action in actions} == {
        "lightweight+metadata",
        "balanced+metadata",
        "balanced+roi",
        "high_accuracy+roi_context",
        "delay/store",
    }
