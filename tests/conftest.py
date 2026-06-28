"""pytest 全局配置：确保仓库根目录在 sys.path 上，使 `import src.xxx` 可用。"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
