"""Shared pytest config + markers for the voxkit test suite."""

from __future__ import annotations

import shutil

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_whisper: skip if whisper-cli or a ggml model is not installed locally",
    )


def _has_whisper() -> bool:
    """Return True iff `whisper-cli` is on PATH AND at least one usable model is findable.

    Cheap pre-filter: rely on the canonical discovery helper from
    ``voxkit.core.whisper_exec`` so we don't duplicate path-search logic.
    Falls back to False on any ImportError so test collection never crashes
    on incomplete installs.
    """
    if not shutil.which("whisper-cli"):
        return False
    try:
        from voxkit.core.whisper_exec import find_whisper_model
    except ImportError:
        return False
    return (
        find_whisper_model("base") is not None
        or find_whisper_model("large-v3-turbo") is not None
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _has_whisper():
        return
    skip_marker = pytest.mark.skip(
        reason="whisper-cli or a ggml model is not installed locally"
    )
    for item in items:
        if "requires_whisper" in item.keywords:
            item.add_marker(skip_marker)
