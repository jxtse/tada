from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


def _resolve_ta_path() -> Path:
    """Locate the paper-facing TADASkill implementation."""
    configured = os.environ.get("TADA_SKILL_ROOT", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())

    repository_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repository_root / "TADASkill",
            repository_root / "TADA_skill" / "TADASkill",
        ]
    )

    for root in candidates:
        ta_path = root / "lib" / "ta.py"
        if ta_path.is_file():
            return ta_path

    searched = ", ".join(str(path / "lib" / "ta.py") for path in candidates)
    raise RuntimeError(
        "Cannot locate TADASkill. Set TADA_SKILL_ROOT to its directory. "
        f"Searched: {searched}"
    )


@lru_cache(maxsize=1)
def _runtime() -> ModuleType:
    """Load the canonical atom runtime once, after endpoint env vars are set."""
    ta_path = _resolve_ta_path()
    module_name = "_tada_skill_runtime"
    spec = importlib.util.spec_from_file_location(module_name, ta_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load TADASkill runtime from {ta_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TextAnalysis:
    """Thin compatibility adapter over the canonical TADASkill atoms.

    Large-scale batching, concurrency, retry/split behavior, joint-tagging
    routing, and taxonomy consolidation remain inside ``ta.tag`` and
    ``ta.categorize``. The agent only composes these two public operations.
    """

    def Tagging(
        self, text_list: list[str], column_name: str, query: str
    ) -> list[Any]:
        return _runtime().tag(text_list, column_name, query)

    def Categorization(
        self, text_list: list[str], column_name: str, query: str
    ) -> list[dict[str, str]]:
        return _runtime().categorize(text_list, column_name, query)
