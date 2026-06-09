from src.common.types import CommState, LinkTrace, ResourceState, TaskSample
from src.context.context_builder import ContextBuilder


def test_context_vector_is_9d():
    task = TaskSample("task_0000", "a", "b", "scene", 0, 0.8, 40.0, 0.5, 0.4)
    resource_state = ResourceState(0.7, 0.9, 0.2)
    comm_state = CommState(8.0, 64.0)
    link_trace = LinkTrace(0, True, 10.0, 6.0, 0.8, 20.0)
    context = ContextBuilder(queue_max_mb=64.0).build(task, resource_state, comm_state, link_trace)
    assert context.as_array().shape == (9,)
