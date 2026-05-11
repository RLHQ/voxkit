"""Root-level voxkit CLI behavior."""

from __future__ import annotations

import pytest

from voxkit import __version__
from voxkit.cli import _build_parser


def test_version_flag_prints_package_version(capsys):
    parser = _build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out == f"voxkit {__version__}\n"
