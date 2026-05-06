#!/usr/bin/env python3
"""
Convenience runner for Migration 001 (additive).

Defaults to using `config.yaml` -> database.path when available.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

def _load_migration_001():
    """
    Load `migrations/001_additive.py` by file path.
    (The leading digits make it an invalid Python module name for direct import.)
    """
    repo_root = Path(__file__).resolve().parents[1]
    migration_path = repo_root / "migrations" / "001_additive.py"
    spec = importlib.util.spec_from_file_location("migration_001_additive", migration_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load migration module from {migration_path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure module is registered so dataclasses (and others) can resolve __module__.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _get_db_path_from_config(config_path: Path) -> str | None:
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    if not config_path.exists():
        return None

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        db = (data.get("database") or {}).get("path")
        return str(db) if db else None
    except Exception:
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "config.yaml"
    db_path = _get_db_path_from_config(config_path) or "matches.db"

    migration = _load_migration_001()
    result = migration.migrate_001_additive_path(db_path)
    print(f"[migration-001] db={db_path} ok, added_columns={result.added_columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

