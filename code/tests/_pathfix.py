"""Import-time side effect: makes ``code/`` importable as a package root
regardless of the current working directory, so these tests run the same
way from the repo root, from ``code/``, or via ``python -m unittest``.
"""
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
