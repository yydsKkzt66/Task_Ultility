"""可选模型下载脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.download_models import download_optional_models  # noqa: E402


if __name__ == "__main__":
    download_optional_models()
