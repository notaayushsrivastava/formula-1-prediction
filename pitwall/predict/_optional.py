"""Optional heavy libraries (pyarrow, lightgbm, torch...) are imported only when needed."""
from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, purpose: str, extra: str = "ml", hint: str = "") -> ModuleType:
    """Import `module`, or raise an ImportError that says exactly how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        message = (f"{purpose} needs the optional package '{module}'. Install the '{extra}' "
                   f'extra: pip install "pitwall-f1[{extra}]"  (from a clone: '
                   f'pip install -e ".[{extra}]").')
        if hint:
            message += f" {hint}"
        raise ImportError(message) from exc
