from src.actions.action_space import ActionSpace


def test_action_space_contains_progressive_payload_actions():
    actions = ActionSpace().list_actions()
    assert len(actions) == 7
    assert {action.action_name for action in actions} == {
        "lightweight+metadata",
        "balanced+metadata",
        "balanced+roi",
        "balanced+metadata_then_roi",
        "high_accuracy+metadata_then_roi_context",
        "high_accuracy+roi_context",
        "delay/store",
    }
    assert {action.payload_plan or action.payload_policy for action in actions} >= {
        "metadata",
        "roi",
        "roi_context",
        "defer",
        "metadata_then_roi",
        "metadata_then_roi_context",
    }
