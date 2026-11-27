"""File-based model registry with atomic version switch.

layout:  <root>/versions/<version>/<app>.json
         <root>/LATEST                -> text file containing the live version

A version is written completely before LATEST is switched with os.replace (atomic
on POSIX), so a reader never sees a half-published set of models. In production
this would be S3 + a pointer object (or a model registry service).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from autoscaler.forecaster import Artifact


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write(self, version: str, art: Artifact) -> Path:
        d = self.root / "versions" / version
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{art.app}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(art.to_dict()))
        os.replace(tmp, p)
        return p

    def publish(self, version: str, expected_apps: list[str] | None = None) -> None:
        d = self.root / "versions" / version
        present = {p.stem for p in d.glob("*.json")}
        if expected_apps is not None and set(expected_apps) - present:
            raise RuntimeError(f"version {version} incomplete: missing "
                               f"{sorted(set(expected_apps) - present)}")
        tmp = self.root / "LATEST.tmp"
        tmp.write_text(version)
        os.replace(tmp, self.root / "LATEST")

    def latest_version(self) -> str | None:
        p = self.root / "LATEST"
        return p.read_text().strip() if p.exists() else None

    def load(self, version: str | None = None) -> dict[str, Artifact]:
        version = version or self.latest_version()
        if version is None:
            return {}
        d = self.root / "versions" / version
        return {p.stem: Artifact.from_dict(json.loads(p.read_text())) for p in d.glob("*.json")}
