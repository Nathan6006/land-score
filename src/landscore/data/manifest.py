"""Record every fetched file (source URL, date, checksum) so both machines get identical inputs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from landscore.config import data_dir

MANIFEST_NAME = "manifest.json"


def manifest_path() -> Path:
    return data_dir() / MANIFEST_NAME


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load() -> dict:
    p = manifest_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"files": {}}


def save(m: dict) -> None:
    p = manifest_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(data_dir().resolve()).as_posix()


def record(path: Path, sources: list[str], **meta) -> dict:
    """Add or replace the manifest entry for a local file under the data folder."""
    path = Path(path)
    entry = {
        "sources": list(sources),
        "retrieved": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        **meta,
    }
    m = load()
    m["files"][rel(path)] = entry
    save(m)
    return entry


def is_current(path: Path) -> bool:
    """True if the file exists and matches its manifest checksum (so it can be skipped)."""
    path = Path(path)
    if not path.exists():
        return False
    entry = load()["files"].get(rel(path))
    return bool(entry) and entry["bytes"] == path.stat().st_size and entry["sha256"] == sha256(path)
