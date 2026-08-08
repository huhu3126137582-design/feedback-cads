"""Load flat YAML configurations or named profiles from one YAML bundle."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_yaml_profile(
    path: Path,
    *,
    profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one experiment configuration and its immutable identity metadata.

    Flat legacy YAML files remain supported.  A bundled file must expose a
    ``profiles`` mapping whose entries contain ``config`` and may contain a
    frozen ``legacy_sha256`` and a default ``output_dir``.
    """

    path = path.resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")

    profiles = document.get("profiles")
    if profiles is None:
        if profile is not None:
            raise ValueError(
                f"--profile={profile!r} was supplied for a flat config: {path}"
            )
        return deepcopy(document), {
            "bundle_path": str(path),
            "profile": None,
            "legacy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "output_dir": None,
        }

    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"Configuration bundle has no profiles: {path}")
    if profile is None:
        available = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(
            f"A profile is required for {path.name}; choose one of: {available}"
        )
    if profile not in profiles:
        available = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(
            f"Unknown profile {profile!r} in {path.name}; choose one of: "
            f"{available}"
        )

    entry = profiles[profile]
    if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
        raise ValueError(f"Invalid profile {profile!r} in {path}")
    legacy_sha256 = entry.get("legacy_sha256")
    if not isinstance(legacy_sha256, str) or len(legacy_sha256) != 64:
        raise ValueError(f"Profile {profile!r} lacks a frozen legacy SHA-256.")
    return deepcopy(entry["config"]), {
        "bundle_path": str(path),
        "profile": profile,
        "legacy_sha256": legacy_sha256,
        "output_dir": entry.get("output_dir"),
    }
