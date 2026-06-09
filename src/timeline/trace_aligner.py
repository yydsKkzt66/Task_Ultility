"""trace 对齐辅助函数。"""

from __future__ import annotations


def align_to_length(items: list, target_length: int) -> list:
    """把输入序列对齐到统一时间轴长度。"""

    if len(items) == target_length:
        return list(items)
    if len(items) > target_length:
        return list(items[:target_length])
    if not items:
        raise ValueError("无法对齐空序列。")
    padded = list(items)
    while len(padded) < target_length:
        padded.append(items[-1])
    return padded
