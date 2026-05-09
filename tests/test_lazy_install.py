"""Unit tests for voxkit.core.lazy_install repair paths."""

from __future__ import annotations

from pathlib import Path

from voxkit.core import lazy_install as L


def test_cached_venv_repairs_missing_voxkit_worker(monkeypatch, tmp_path: Path):
    """A pyannote-ready venv must still be repaired if voxkit is not importable."""
    calls: list[str] = []
    fake_py = tmp_path / "venv" / "bin" / "python"
    fake_py.parent.mkdir(parents=True)
    fake_py.write_text("", encoding="utf-8")

    monkeypatch.setattr(L, "VENV_DIR", tmp_path / "venv")
    monkeypatch.setattr(L, "MARKER", tmp_path / ".installed")
    monkeypatch.setattr(L, "_venv_python", lambda venv: fake_py)
    monkeypatch.setattr(
        L,
        "_read_marker",
        lambda: f"pyannote.audio=4.0.4\nspec={L.PYANNOTE_VERSION_SPEC}\n",
    )
    monkeypatch.setattr(L, "_check_pyannote_version", lambda py: "4.0.4")
    monkeypatch.setattr(L, "_have_uv", lambda: "/fake/uv")

    importable = {"ok": False}

    def _fake_check_worker(py: Path) -> bool:
        calls.append("check_worker")
        return importable["ok"]

    def _fake_install_voxkit(uv_bin: str, py: Path, *, verbose: bool) -> None:
        calls.append("install_voxkit")
        importable["ok"] = True

    monkeypatch.setattr(L, "_check_voxkit_worker_importable", _fake_check_worker)
    monkeypatch.setattr(L, "_install_voxkit_package", _fake_install_voxkit)
    monkeypatch.setattr(L, "_install_packages", lambda *a, **kw: calls.append("install_all"))

    info = L.ensure_venv(verbose=False)

    assert info.venv_python == fake_py
    assert info.pyannote_version == "4.0.4"
    assert calls == ["check_worker", "install_voxkit", "check_worker"]


def test_cached_venv_skips_repair_when_worker_importable(monkeypatch, tmp_path: Path):
    fake_py = tmp_path / "venv" / "bin" / "python"
    fake_py.parent.mkdir(parents=True)
    fake_py.write_text("", encoding="utf-8")

    monkeypatch.setattr(L, "VENV_DIR", tmp_path / "venv")
    monkeypatch.setattr(L, "MARKER", tmp_path / ".installed")
    monkeypatch.setattr(L, "_venv_python", lambda venv: fake_py)
    monkeypatch.setattr(
        L,
        "_read_marker",
        lambda: f"pyannote.audio=4.0.4\nspec={L.PYANNOTE_VERSION_SPEC}\n",
    )
    monkeypatch.setattr(L, "_check_pyannote_version", lambda py: "4.0.4")
    monkeypatch.setattr(L, "_check_voxkit_worker_importable", lambda py: True)

    repaired = {"called": False}

    def _unexpected_repair(*args, **kwargs) -> None:
        repaired["called"] = True

    monkeypatch.setattr(L, "_install_voxkit_package", _unexpected_repair)
    monkeypatch.setattr(L, "_install_packages", _unexpected_repair)

    info = L.ensure_venv(verbose=False)

    assert info.venv_python == fake_py
    assert repaired["called"] is False
