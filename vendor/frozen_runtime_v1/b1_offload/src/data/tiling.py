"""图像切片工具占位。"""

from __future__ import annotations

from pathlib import Path


def build_tiles(image_path: Path, tile_size: int, tile_stride: int) -> list[tuple[int, int, int, int]]:
    """占位函数：未来用于真实遥感图像切片。

    当前 mock scaffold 不依赖真实图像，因此这里返回空列表即可。
    """

    _ = (image_path, tile_size, tile_stride)
    return []
