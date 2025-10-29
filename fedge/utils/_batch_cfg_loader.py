from __future__ import annotations
from pathlib import Path

# Python 3.10: prefer tomli; fallback to tomllib if present
try:
    import tomli as tomllib  # type: ignore
except Exception:  # pragma: no cover
    import tomllib  # type: ignore

DEFAULTS = {"client": 32, "server": 32, "cloud": 32}

def _candidates() -> list[Path]:
    # 1) PROJECT_ROOT env
    pr = Path((__import__("os").environ.get("PROJECT_ROOT") or "").strip() or ".").resolve()
    # 2) repo root from this file: fedge/utils/_batch_cfg_loader.py -> repo = parents[2]
    here = Path(__file__).resolve()
    repo = here.parents[2]
    # 3) cwd
    cwd = Path.cwd()
    return [pr / "pyproject.toml", repo / "pyproject.toml", cwd / "pyproject.toml"]

def load() -> dict:
    for p in _candidates():
        try:
            if p.is_file():
                data = tomllib.loads(p.read_text(encoding="utf-8"))
                return data["tool"]["flwr"]["cluster"]["batch_sizes"]
        except KeyError:
            # found file but missing table, try next
            continue
        except Exception:
            continue
    # fallback
    return dict(DEFAULTS)
