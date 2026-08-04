"""项目级 Python 启动定制。

本文件的唯一目的，是在进入 pytest 之前关闭第三方插件自动加载。
这样可以避免本机环境中与项目无关的 pytest 插件拖慢甚至阻塞测试启动，
保证 `pytest` 主要执行仓库自身的测试逻辑。
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
