"""whisper_exec 模块单元测试。

策略：
- discovery / build_argv / parse_whisper_json / PROGRESS_RE 全是纯函数，直接断言。
- ``run_whisper`` 用 monkeypatch 替换 ``subprocess.Popen`` + ``time.monotonic``
  模拟子进程行为，避免依赖真实 whisper-cli。
- 真正调用 whisper-cli 的集成测试用 ``pytest.skipif`` 守门。
"""

from __future__ import annotations

import io
import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Iterable

import pytest

from voxkit.core import whisper_exec as W
from voxkit.core.whisper_exec import (
    CJK_LANGUAGES,
    PROGRESS_RE,
    WhisperFailed,
    WhisperFlags,
    WhisperRunResult,
    WhisperTimeout,
    build_argv,
    find_vad_model,
    find_whisper_cli,
    find_whisper_model,
    parse_whisper_json,
    run_whisper,
)


# ── helpers ────────────────────────────────────────────────────────────
def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _isolate_env(monkeypatch: pytest.MonkeyPatch, home: Path | None = None) -> None:
    """清掉所有可能干扰 discovery 的环境变量；可选地隔离 ``$HOME``。

    传 ``home`` → 把 ``$HOME`` 重定向到指定目录，让 ``Path.home()`` /
    ``Path("~/...").expanduser()`` 落进 tmp 而非开发者真实 home。
    需要这层隔离的场景：``find_vad_model`` / ``find_whisper_model`` 会查
    ``~/.cache/voxkit/{aux,models}/...``，开发者机器上这些路径常驻文件，
    不隔离会让"应当返回 None"的测试意外命中。
    """
    for k in (
        "WHISPER_BIN",
        "WHISPER_MODEL_PATH",
        "WHISPER_VAD_MODEL_PATH",
    ):
        monkeypatch.delenv(k, raising=False)
    if home is not None:
        monkeypatch.setenv("HOME", str(home))


# ───────────────────────────────────────────────────────────────────────
# find_whisper_cli
# ───────────────────────────────────────────────────────────────────────
def test_find_whisper_cli_override_existing(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setattr(W.shutil, "which", lambda _name: None)
    fake = _make_executable(tmp_path / "whisper-cli")
    got = find_whisper_cli(override=fake)
    assert got == fake


def test_find_whisper_cli_override_missing_falls_through(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setattr(W.shutil, "which", lambda _name: None)
    # 屏蔽 brew fallback，避免在 dev 机命中真实安装
    monkeypatch.setattr(W, "_is_executable", lambda p: False)
    # override 指向不存在路径，应当继续 fallback；fallback 也都没命中 → None
    got = find_whisper_cli(override=tmp_path / "nope")
    assert got is None


def test_find_whisper_cli_env_bin(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setattr(W.shutil, "which", lambda _name: None)
    fake = _make_executable(tmp_path / "envbin")
    monkeypatch.setenv("WHISPER_BIN", str(fake))
    got = find_whisper_cli()
    assert got == fake


def test_find_whisper_cli_which(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    fake = _make_executable(tmp_path / "wc-which")
    monkeypatch.setattr(W.shutil, "which", lambda name: str(fake) if name == "whisper-cli" else None)
    got = find_whisper_cli()
    assert got == fake


def test_find_whisper_cli_none(monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setattr(W.shutil, "which", lambda _name: None)
    # 屏蔽 brew 路径：用一个不存在的临时目录覆盖不太现实；直接 mock _is_executable
    monkeypatch.setattr(W, "_is_executable", lambda p: False)
    got = find_whisper_cli()
    assert got is None


# ───────────────────────────────────────────────────────────────────────
# find_whisper_model
# ───────────────────────────────────────────────────────────────────────
def test_find_whisper_model_override(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    fake = tmp_path / "model.bin"
    fake.write_bytes(b"x")
    got = find_whisper_model(override=fake)
    assert got == fake


def test_find_whisper_model_alias_resolution_user_cache(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    home = tmp_path / "fakehome"
    monkeypatch.setattr(Path, "home", lambda: home)
    cache_dir = home / ".cache" / "voxkit" / "models"
    cache_dir.mkdir(parents=True)
    target = cache_dir / "ggml-large-v3-turbo.bin"
    target.write_bytes(b"x")
    got = find_whisper_model("large-v3-turbo")
    assert got == target


def test_find_whisper_model_quantized_alias(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    home = tmp_path / "fakehome"
    monkeypatch.setattr(Path, "home", lambda: home)
    cache_dir = home / ".cache" / "voxkit" / "models"
    cache_dir.mkdir(parents=True)
    target = cache_dir / "ggml-large-v3-turbo-q5_0.bin"
    target.write_bytes(b"x")
    got = find_whisper_model("q5_0")
    assert got == target


def test_find_whisper_model_env_var(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    home = tmp_path / "fakehome"
    monkeypatch.setattr(Path, "home", lambda: home)
    target = tmp_path / "custom.bin"
    target.write_bytes(b"x")
    monkeypatch.setenv("WHISPER_MODEL_PATH", str(target))
    got = find_whisper_model("medium")
    assert got == target


def test_find_whisper_model_none(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    home = tmp_path / "fakehome"
    monkeypatch.setattr(Path, "home", lambda: home)
    # brew 默认目录也不会存在（在 tmp_path 外，但我们 mock Path.is_file 简单点）
    real_is_file = Path.is_file

    def fake_is_file(self):  # type: ignore[override]
        # 屏蔽所有 absolute path 为 /opt/homebrew/... 的命中
        if str(self).startswith("/opt/homebrew/"):
            return False
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    got = find_whisper_model("nonexistent-model")
    assert got is None


def test_find_whisper_model_absolute_path_as_name(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    target = tmp_path / "weights.bin"
    target.write_bytes(b"x")
    got = find_whisper_model(str(target))
    assert got == target


# ───────────────────────────────────────────────────────────────────────
# find_vad_model
# ───────────────────────────────────────────────────────────────────────
def test_find_vad_model_env(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    target = tmp_path / "vad.bin"
    target.write_bytes(b"x")
    monkeypatch.setenv("WHISPER_VAD_MODEL_PATH", str(target))
    got = find_vad_model()
    assert got == target


def test_find_vad_model_override(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    target = tmp_path / "v.bin"
    target.write_bytes(b"x")
    got = find_vad_model(override=target)
    assert got == target


def test_find_vad_model_brew(tmp_path, monkeypatch):
    # 隔离 HOME 让 voxkit aux 路径解析到 tmp（不会命中开发者本机的 ~/.cache/voxkit/aux/）
    _isolate_env(monkeypatch, home=tmp_path)
    real_is_file = Path.is_file

    def fake_is_file(self):  # type: ignore[override]
        if str(self) == "/opt/homebrew/share/whisper-cpp/ggml-silero-v5.1.2.bin":
            return True
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    got = find_vad_model()
    assert got is not None
    assert str(got) == "/opt/homebrew/share/whisper-cpp/ggml-silero-v5.1.2.bin"


def test_find_vad_model_none(tmp_path, monkeypatch):
    _isolate_env(monkeypatch, home=tmp_path)
    real_is_file = Path.is_file

    def fake_is_file(self):  # type: ignore[override]
        if str(self).startswith("/opt/homebrew/"):
            return False
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    got = find_vad_model()
    assert got is None


def test_find_vad_model_voxkit_aux(tmp_path, monkeypatch):
    """voxkit fetch-bundle 装到 ``~/.cache/voxkit/aux/`` 的 silero VAD 应被找到。

    回归保护：之前 ``find_vad_model`` 不查这个路径，导致 fetch-bundle 装的 aux
    在没 brew 的环境（Linux apt / 旧 brew formula）下完全没用。
    """
    _isolate_env(monkeypatch, home=tmp_path)
    aux_path = tmp_path / ".cache" / "voxkit" / "aux" / "ggml-silero-v5.1.2.bin"
    aux_path.parent.mkdir(parents=True)
    aux_path.write_bytes(b"silero-fake")

    got = find_vad_model()
    assert got == aux_path


def test_find_vad_model_voxkit_aux_priority_over_brew(tmp_path, monkeypatch):
    """voxkit aux 与 brew 都在场时，优先 voxkit aux（受 voxkit manifest 校验保护，更可控）。"""
    _isolate_env(monkeypatch, home=tmp_path)

    aux_path = tmp_path / ".cache" / "voxkit" / "aux" / "ggml-silero-v5.1.2.bin"
    aux_path.parent.mkdir(parents=True)
    aux_path.write_bytes(b"voxkit-aux-version")

    real_is_file = Path.is_file

    def fake_is_file(self):  # type: ignore[override]
        if str(self) == "/opt/homebrew/share/whisper-cpp/ggml-silero-v5.1.2.bin":
            return True
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    got = find_vad_model()
    assert got == aux_path, "voxkit aux 应优先于 brew 路径"


# ───────────────────────────────────────────────────────────────────────
# build_argv
# ───────────────────────────────────────────────────────────────────────
def _flags(**overrides) -> WhisperFlags:
    base = dict(
        model_path=Path("/m.bin"),
        language="en",
        vad=True,
        vad_model_path=Path("/vad.bin"),
        logprob_thold=-0.8,
        word_timestamps=True,
        max_context_zero=True,
        threads=None,
        extra=[],
    )
    base.update(overrides)
    return WhisperFlags(**base)


def _argv_pairs(argv: list[str]) -> list[tuple[str, str | None]]:
    """便于断言：把 argv 拍成 (flag, next-token-or-None)。"""
    out: list[tuple[str, str | None]] = []
    for i, tok in enumerate(argv):
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        out.append((tok, nxt))
    return out


def test_build_argv_english_word_timestamps_vad():
    argv = build_argv(
        _flags(language="en"),
        Path("/a.wav"),
        Path("/o.json"),
        whisper_bin=Path("/wc"),
    )
    # 关键 flag 必须出现
    assert "--max-context" in argv
    assert "--logprob-thold" in argv
    assert "--max-len" in argv
    assert "--split-on-word" in argv
    assert "--vad" in argv
    assert "--vad-model" in argv
    assert "-ojf" in argv
    assert "--print-progress" in argv
    assert "--no-prints" in argv

    # 关键值
    pairs = _argv_pairs(argv)
    assert ("--max-context", "0") in pairs
    assert ("--logprob-thold", "-0.8") in pairs
    assert ("--max-len", "1") in pairs
    assert ("--vad-model", "/vad.bin") in pairs
    assert ("-l", "en") in pairs
    assert ("-m", "/m.bin") in pairs
    assert ("-f", "/a.wav") in pairs

    # -of 必须是 prefix（无 .json 后缀）
    assert ("-of", "/o") in pairs


@pytest.mark.parametrize("lang", sorted(CJK_LANGUAGES))
def test_build_argv_cjk_drops_word_timestamp_flags(lang):
    argv = build_argv(
        _flags(language=lang, word_timestamps=True),
        Path("/a.wav"),
        Path("/o.json"),
        whisper_bin=Path("/wc"),
    )
    # CJK 一定不带 word-timestamp 强制 1-token flags
    assert "--max-len" not in argv
    assert "--split-on-word" not in argv
    # 但 anti-hallucination 仍在
    assert "--max-context" in argv
    assert "--logprob-thold" in argv
    # VAD 仍在
    assert "--vad" in argv


def test_build_argv_vad_true_but_no_model_path():
    argv = build_argv(
        _flags(vad=True, vad_model_path=None),
        Path("/a.wav"),
        Path("/o.json"),
        whisper_bin=Path("/wc"),
    )
    assert "--vad" not in argv
    assert "--vad-model" not in argv


def test_build_argv_extra_appended_at_end():
    argv = build_argv(
        _flags(extra=["-bs", "5"]),
        Path("/a.wav"),
        Path("/o.json"),
        whisper_bin=Path("/wc"),
    )
    assert argv[-2:] == ["-bs", "5"]


def test_build_argv_strips_json_suffix():
    argv = build_argv(
        _flags(),
        Path("/a.wav"),
        Path("/tmp/foo.json"),
        whisper_bin=Path("/wc"),
    )
    pairs = _argv_pairs(argv)
    assert ("-of", "/tmp/foo") in pairs


def test_build_argv_threads():
    argv = build_argv(
        _flags(threads=8),
        Path("/a.wav"),
        Path("/o.json"),
        whisper_bin=Path("/wc"),
    )
    pairs = _argv_pairs(argv)
    assert ("--threads", "8") in pairs


def test_build_argv_max_context_off():
    argv = build_argv(
        _flags(max_context_zero=False),
        Path("/a.wav"),
        Path("/o.json"),
        whisper_bin=Path("/wc"),
    )
    assert "--max-context" not in argv


def test_build_argv_word_timestamps_off_english():
    argv = build_argv(
        _flags(language="en", word_timestamps=False),
        Path("/a.wav"),
        Path("/o.json"),
        whisper_bin=Path("/wc"),
    )
    assert "--max-len" not in argv
    assert "--split-on-word" not in argv


def test_build_argv_starts_with_whisper_bin():
    argv = build_argv(
        _flags(),
        Path("/a.wav"),
        Path("/o.json"),
        whisper_bin=Path("/usr/local/bin/whisper-cli"),
    )
    assert argv[0] == "/usr/local/bin/whisper-cli"


# ───────────────────────────────────────────────────────────────────────
# parse_whisper_json
# ───────────────────────────────────────────────────────────────────────
def test_parse_whisper_json_empty_transcription():
    assert parse_whisper_json({"transcription": []}) == []


def test_parse_whisper_json_missing_transcription_key():
    assert parse_whisper_json({}) == []


def test_parse_whisper_json_single_entry():
    raw = {
        "transcription": [
            {
                "text": " Hello",
                "offsets": {"from": 100, "to": 300},
                "no_speech_prob": 0.05,
            }
        ]
    }
    out = parse_whisper_json(raw)
    assert len(out) == 1
    e = out[0]
    assert e.text == " Hello"
    assert e.t_from_ms == 100
    assert e.t_to_ms == 300
    assert e.no_speech_prob == pytest.approx(0.05)
    assert e.raw == raw["transcription"][0]


def test_parse_whisper_json_filters_empty_text():
    raw = {
        "transcription": [
            {"text": "", "offsets": {"from": 0, "to": 10}},
            {"text": "   ", "offsets": {"from": 10, "to": 20}},
            {"text": " Hi", "offsets": {"from": 20, "to": 40}},
        ]
    }
    out = parse_whisper_json(raw)
    assert len(out) == 1
    assert out[0].text == " Hi"


def test_parse_whisper_json_filters_meta_tokens():
    raw = {
        "transcription": [
            {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}},
            {"text": "[_TT_5]", "offsets": {"from": 0, "to": 0}},
            {"text": " hello", "offsets": {"from": 0, "to": 100}},
        ]
    }
    out = parse_whisper_json(raw)
    assert len(out) == 1
    assert out[0].text == " hello"


def test_parse_whisper_json_no_speech_prob_optional():
    raw = {
        "transcription": [
            {"text": "a", "offsets": {"from": 0, "to": 10}},
            {"text": "b", "offsets": {"from": 10, "to": 20}, "no_speech_prob": 0.9},
        ]
    }
    out = parse_whisper_json(raw)
    assert out[0].no_speech_prob is None
    assert out[1].no_speech_prob == pytest.approx(0.9)


def test_parse_whisper_json_preserves_order():
    raw = {
        "transcription": [
            {"text": "a", "offsets": {"from": 0, "to": 1}},
            {"text": "b", "offsets": {"from": 1, "to": 2}},
            {"text": "c", "offsets": {"from": 2, "to": 3}},
        ]
    }
    out = parse_whisper_json(raw)
    assert [e.text for e in out] == ["a", "b", "c"]


# ───────────────────────────────────────────────────────────────────────
# PROGRESS_RE
# ───────────────────────────────────────────────────────────────────────
def test_progress_re_with_spaces():
    m = PROGRESS_RE.search(b"whisper_print_progress_callback: progress =  5%")
    assert m is not None
    assert m.group(1) == b"5"


def test_progress_re_no_spaces():
    m = PROGRESS_RE.search(b"progress=42%")
    assert m is not None
    assert m.group(1) == b"42"


def test_progress_re_no_match():
    assert PROGRESS_RE.search(b"some other line") is None


# ───────────────────────────────────────────────────────────────────────
# run_whisper — mocked subprocess
# ───────────────────────────────────────────────────────────────────────
class _FakeStream:
    """模拟 subprocess.PIPE 的可 readline 流。"""

    def __init__(self, lines: Iterable[bytes]):
        self._buf = io.BytesIO(b"".join(lines))

    def readline(self) -> bytes:
        return self._buf.readline()

    def read(self) -> bytes:
        return self._buf.read()

    def close(self) -> None:
        self._buf.close()


class _FakePopen:
    """主线程用 ``poll()`` 看终止；后台线程读 stderr。

    我们让 ``poll()`` 在 stderr 流被读完后返回 ``returncode``，模拟自然结束。
    """

    def __init__(
        self,
        argv,
        *,
        stderr_lines: list[bytes],
        returncode: int = 0,
        json_payload: dict | None = None,
        out_json_path: Path | None = None,
        hang: bool = False,
        **kwargs,
    ):
        self.argv = argv
        self.kwargs = kwargs
        self.stderr = _FakeStream(stderr_lines)
        self._returncode = returncode
        self._hang = hang
        self._killed = False
        self._waited = False
        # 写入 JSON 文件，模拟 whisper-cli 行为
        if json_payload is not None and out_json_path is not None:
            out_json_path.parent.mkdir(parents=True, exist_ok=True)
            out_json_path.write_text(json.dumps(json_payload), encoding="utf-8")

    @property
    def returncode(self):
        return self._returncode if not self._hang or self._killed else None

    def poll(self):
        if self._hang and not self._killed:
            return None
        return self._returncode

    def wait(self, timeout=None):
        self._waited = True
        return self._returncode

    def kill(self):
        self._killed = True


def test_run_whisper_happy_path(tmp_path, monkeypatch):
    out_json = tmp_path / "out.json"
    json_path = tmp_path / "out.json"  # build_argv prefix → ".json"

    progress_lines = [
        b"whisper_print_progress_callback: progress =  5%\n",
        b"whisper_print_progress_callback: progress = 25%\n",
        b"whisper_print_progress_callback: progress = 50%\n",
        b"whisper_print_progress_callback: progress = 75%\n",
        b"whisper_print_progress_callback: progress = 100%\n",
        b"some non-progress line\n",
    ]
    payload = {
        "transcription": [
            {"text": " Hi", "offsets": {"from": 0, "to": 200}},
            {"text": " there", "offsets": {"from": 200, "to": 500}},
        ]
    }

    def fake_popen(argv, **kwargs):
        return _FakePopen(
            argv,
            stderr_lines=progress_lines,
            returncode=0,
            json_payload=payload,
            out_json_path=json_path,
            **kwargs,
        )

    monkeypatch.setattr(W.subprocess, "Popen", fake_popen)

    seen: list[int] = []
    flags = _flags(language="en")
    result = run_whisper(
        audio=tmp_path / "a.wav",
        out_json=out_json,
        flags=flags,
        whisper_bin=Path("/wc"),
        timeout_secs=30.0,
        progress_cb=seen.append,
    )

    assert seen == [5, 25, 50, 75, 100]
    assert isinstance(result, WhisperRunResult)
    assert result.elapsed_secs >= 0.0
    assert len(result.entries) == 2
    assert result.entries[0].text == " Hi"
    assert result.raw_json == payload


def test_run_whisper_progress_dedup(tmp_path, monkeypatch):
    """同一百分比连续出现只回调一次。"""
    out_json = tmp_path / "out.json"
    progress_lines = [
        b"progress = 10%\n",
        b"progress = 10%\n",
        b"progress = 10%\n",
        b"progress = 50%\n",
    ]
    payload = {"transcription": []}

    def fake_popen(argv, **kwargs):
        return _FakePopen(
            argv,
            stderr_lines=progress_lines,
            returncode=0,
            json_payload=payload,
            out_json_path=out_json,
            **kwargs,
        )

    monkeypatch.setattr(W.subprocess, "Popen", fake_popen)

    seen: list[int] = []
    run_whisper(
        audio=tmp_path / "a.wav",
        out_json=out_json,
        flags=_flags(),
        whisper_bin=Path("/wc"),
        timeout_secs=10.0,
        progress_cb=seen.append,
    )
    assert seen == [10, 50]


def test_run_whisper_nonzero_exit(tmp_path, monkeypatch):
    out_json = tmp_path / "out.json"
    stderr_lines = [b"some error happened\n"] * 60  # >50 行检验 tail 截断

    def fake_popen(argv, **kwargs):
        # 不写 JSON 文件
        return _FakePopen(
            argv,
            stderr_lines=stderr_lines,
            returncode=2,
            json_payload=None,
            out_json_path=None,
            **kwargs,
        )

    monkeypatch.setattr(W.subprocess, "Popen", fake_popen)

    with pytest.raises(WhisperFailed) as exc_info:
        run_whisper(
            audio=tmp_path / "a.wav",
            out_json=out_json,
            flags=_flags(),
            whisper_bin=Path("/wc"),
            timeout_secs=5.0,
        )
    err = exc_info.value
    assert err.returncode == 2
    assert "some error happened" in err.stderr_tail
    # tail 最多 50 行
    assert err.stderr_tail.count("some error happened") <= 50


def test_run_whisper_timeout(tmp_path, monkeypatch):
    out_json = tmp_path / "out.json"

    def fake_popen(argv, **kwargs):
        return _FakePopen(
            argv,
            stderr_lines=[],
            returncode=0,
            json_payload=None,
            out_json_path=None,
            hang=True,  # poll 永远返回 None
            **kwargs,
        )

    monkeypatch.setattr(W.subprocess, "Popen", fake_popen)

    # 用极小的 timeout 让循环立刻命中 deadline
    with pytest.raises(WhisperTimeout):
        run_whisper(
            audio=tmp_path / "a.wav",
            out_json=out_json,
            flags=_flags(),
            whisper_bin=Path("/wc"),
            timeout_secs=0.1,
        )


def test_run_whisper_missing_json_after_zero_exit(tmp_path, monkeypatch):
    """returncode=0 但 JSON 文件不存在 → 升级为 WhisperFailed。"""
    out_json = tmp_path / "out.json"

    def fake_popen(argv, **kwargs):
        return _FakePopen(
            argv,
            stderr_lines=[b"weird\n"],
            returncode=0,
            json_payload=None,
            out_json_path=None,
            **kwargs,
        )

    monkeypatch.setattr(W.subprocess, "Popen", fake_popen)

    with pytest.raises(WhisperFailed):
        run_whisper(
            audio=tmp_path / "a.wav",
            out_json=out_json,
            flags=_flags(),
            whisper_bin=Path("/wc"),
            timeout_secs=5.0,
        )


def test_run_whisper_handles_invalid_utf8_in_token_text(tmp_path, monkeypatch):
    """Regression: whisper.cpp ``--output-json-full`` splits CJK characters
    at byte boundaries inside ``tokens[].text``, so the resulting JSON file
    contains invalid UTF-8 byte sequences. Strict utf-8 decode crashed the
    pipeline on Chinese audio. Decoding with ``errors="replace"`` lets json
    parse normally; segment-level ``text`` (the only field voxkit consumes)
    is well-formed and unaffected."""
    out_json = tmp_path / "out.json"
    json_path = tmp_path / "out.json"

    # 手工构造：segment.text 是合法 UTF-8（"去做"），tokens[].text 含一个被
    # 切到字节中间的 3 字节字符（\xe9\x80 是前 2 字节，\x89 是第 3 字节）。
    bad_json = (
        b'{\n'
        b'  "transcription": [\n'
        b'    {\n'
        b'      "text": "\xe5\x8e\xbb\xe5\x81\x9a",\n'
        b'      "offsets": {"from": 0, "to": 200},\n'
        b'      "tokens": [\n'
        b'        {"text": "\xe9\x80", "offsets": {"from": 0, "to": 100}},\n'
        b'        {"text": "\x89", "offsets": {"from": 100, "to": 200}}\n'
        b'      ]\n'
        b'    }\n'
        b'  ]\n'
        b'}\n'
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(bad_json)

    def fake_popen(argv, **kwargs):
        # json_payload=None → _FakePopen 不会覆盖我们写好的 bad_json
        return _FakePopen(
            argv,
            stderr_lines=[b"progress = 100%\n"],
            returncode=0,
            json_payload=None,
            out_json_path=None,
            **kwargs,
        )

    monkeypatch.setattr(W.subprocess, "Popen", fake_popen)

    flags = _flags(language="zh", word_timestamps=False)
    result = run_whisper(
        audio=tmp_path / "a.wav",
        out_json=out_json,
        flags=flags,
        whisper_bin=Path("/wc"),
        timeout_secs=5.0,
    )

    # 不抛 UnicodeDecodeError；segment 级文本完好
    assert isinstance(result, WhisperRunResult)
    assert len(result.entries) == 1
    assert result.entries[0].text == "去做"


def test_run_whisper_progress_cb_exception_does_not_fail(tmp_path, monkeypatch):
    """progress_cb 抛异常不应中断转写。"""
    out_json = tmp_path / "out.json"
    payload = {"transcription": []}

    def fake_popen(argv, **kwargs):
        return _FakePopen(
            argv,
            stderr_lines=[b"progress = 50%\n"],
            returncode=0,
            json_payload=payload,
            out_json_path=out_json,
            **kwargs,
        )

    monkeypatch.setattr(W.subprocess, "Popen", fake_popen)

    def boom(_pct):
        raise RuntimeError("boom")

    result = run_whisper(
        audio=tmp_path / "a.wav",
        out_json=out_json,
        flags=_flags(),
        whisper_bin=Path("/wc"),
        timeout_secs=5.0,
        progress_cb=boom,
    )
    assert isinstance(result, WhisperRunResult)


# ───────────────────────────────────────────────────────────────────────
# 集成测试（仅当 whisper-cli + 模型都可用）
# ───────────────────────────────────────────────────────────────────────
_HAVE_WHISPER = find_whisper_cli() is not None
_HAVE_MODEL = (
    find_whisper_model("base") is not None
    or find_whisper_model("large-v3-turbo") is not None
)
_FIXTURE = Path(__file__).parent / "fixtures" / "short.wav"


@pytest.mark.skipif(
    not (_HAVE_WHISPER and _HAVE_MODEL and _FIXTURE.is_file()),
    reason="whisper-cli, ggml model, or fixture audio not available",
)
def test_run_whisper_integration_smoke(tmp_path):
    bin_path = find_whisper_cli()
    model = find_whisper_model("base") or find_whisper_model("large-v3-turbo")
    assert bin_path is not None and model is not None  # mypy

    flags = WhisperFlags(
        model_path=model,
        language="en",
        vad=False,
        vad_model_path=None,
        word_timestamps=False,  # 加快、减少输出量
    )
    out = tmp_path / "out.json"
    result = run_whisper(
        audio=_FIXTURE,
        out_json=out,
        flags=flags,
        whisper_bin=bin_path,
        timeout_secs=120.0,
    )
    assert isinstance(result.entries, list)
    assert result.elapsed_secs > 0
