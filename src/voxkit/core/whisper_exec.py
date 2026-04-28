"""whisper.cpp ``whisper-cli`` 单 chunk 调用层。

调用方（pipeline）负责切分；本模块只关心：

- 二进制 / 模型 / VAD 模型的发现（discovery）
- argv 构造（纯函数，便于测试）
- subprocess 拉起 + 进度流式解析 + JSON 解析
- 错误归类（``WhisperFailed`` / ``WhisperTimeout``）

本模块不持有任何 chunk 切分逻辑；上层 pipeline 自行决定何时调用 ``run_whisper``。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from voxkit.core.constants import CJK_LANGUAGES
from voxkit.core.types import Entry

# ── stderr 进度行正则（覆盖 ``progress = 5%`` / ``progress=42%`` 两种）─
PROGRESS_RE = re.compile(rb"progress\s*=\s*(\d+)\s*%")


# ── 异常 ────────────────────────────────────────────────────────────────
class WhisperFailed(RuntimeError):
    """whisper-cli 返回非零退出码。

    Attributes:
        returncode: subprocess 返回码。
        stderr_tail: stderr 末尾若干行（最多 50 行），用于排错。
    """

    def __init__(self, returncode: int, stderr_tail: str):
        super().__init__(
            f"whisper-cli failed with returncode={returncode}\n"
            f"--- stderr tail ---\n{stderr_tail}"
        )
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class WhisperTimeout(RuntimeError):
    """whisper-cli 在 ``timeout_secs`` 内未返回。"""


# ── Discovery ──────────────────────────────────────────────────────────
def _is_executable(path: Path) -> bool:
    """Path 存在且是文件且可执行。"""
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def find_whisper_cli(*, override: Path | None = None) -> Path | None:
    """按优先级查找 whisper-cli 可执行文件。

    Order:
      1. ``override``（若可执行）
      2. ``$WHISPER_BIN`` 环境变量（若可执行）
      3. ``shutil.which("whisper-cli")``
      4. ``/opt/homebrew/bin/whisper-cli``
      5. ``/usr/local/bin/whisper-cli``

    Returns:
        命中第一个的 Path；都没找到返回 None。
    """
    if override is not None:
        p = Path(override)
        if _is_executable(p):
            return p

    env_bin = os.environ.get("WHISPER_BIN")
    if env_bin:
        p = Path(env_bin)
        if _is_executable(p):
            return p

    which = shutil.which("whisper-cli")
    if which:
        return Path(which)

    for candidate in (
        Path("/opt/homebrew/bin/whisper-cli"),
        Path("/usr/local/bin/whisper-cli"),
    ):
        if _is_executable(candidate):
            return candidate

    return None


def find_whisper_model(
    name: str = "large-v3-turbo",
    *,
    override: Path | None = None,
) -> Path | None:
    """按优先级查找 whisper.cpp ggml 模型文件。

    ``name`` 既可以是别名（``large-v3-turbo`` / ``q5_0`` 即
    ``large-v3-turbo-q5_0`` / ``medium`` 等），也可以是绝对/已存在的路径。

    Order:
      1. ``override``（按路径校验存在性）
      2. ``name`` 本身是绝对路径且文件存在 → 直接返回
      3. ``~/.cache/voxkit/models/ggml-{name}.bin``
      4. ``$WHISPER_MODEL_PATH`` 环境变量（视为完整路径）
      5. ``/opt/homebrew/share/whisper-cpp/ggml-{name}.bin``

    Returns:
        命中的 Path；都没找到返回 None。
    """
    if override is not None:
        p = Path(override)
        if p.is_file():
            return p

    # name 本身是绝对路径
    name_path = Path(name)
    if name_path.is_absolute() and name_path.is_file():
        return name_path

    # ``q5_0`` 之类裸量化后缀 → 拼回到 large-v3-turbo-{suffix}
    # 简单约定：以 ``q`` 开头且仅含小写字母+数字+下划线视为量化别名
    resolved_name = name
    if re.fullmatch(r"q[0-9a-z_]+", name):
        resolved_name = f"large-v3-turbo-{name}"

    # 用户级 cache
    user_cache = (
        Path.home() / ".cache" / "voxkit" / "models" / f"ggml-{resolved_name}.bin"
    )
    if user_cache.is_file():
        return user_cache

    env_model = os.environ.get("WHISPER_MODEL_PATH")
    if env_model:
        p = Path(env_model)
        if p.is_file():
            return p

    brew_share = Path(
        f"/opt/homebrew/share/whisper-cpp/ggml-{resolved_name}.bin"
    )
    if brew_share.is_file():
        return brew_share

    return None


def find_vad_model(*, override: Path | None = None) -> Path | None:
    """按优先级查找 silero VAD 模型。

    Order:
      1. ``override``
      2. ``$WHISPER_VAD_MODEL_PATH`` 环境变量
      3. ``/opt/homebrew/share/whisper-cpp/ggml-silero-v5.1.2.bin``

    Returns:
        Path（若存在）或 None（caller 可以决定是否禁用 VAD 并 warn-once）。
    """
    if override is not None:
        p = Path(override)
        if p.is_file():
            return p

    env_vad = os.environ.get("WHISPER_VAD_MODEL_PATH")
    if env_vad:
        p = Path(env_vad)
        if p.is_file():
            return p

    brew = Path("/opt/homebrew/share/whisper-cpp/ggml-silero-v5.1.2.bin")
    if brew.is_file():
        return brew

    return None


# ── Flags ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class WhisperFlags:
    """``whisper-cli`` 的高层参数包。

    Attributes:
        model_path: ggml 模型完整路径。
        language: ``auto`` / ``en`` / ``zh`` / ``ja`` / ``ko`` / ``yue`` / ...。
        vad: 是否启用 VAD（仅当 ``vad_model_path`` 同时给出时才真正传给 cli）。
        vad_model_path: silero VAD 模型路径。``vad=True`` 但路径为 None 时
            两个 flag 都不会出现（防御性默认，由 caller 决定 warn-once）。
        logprob_thold: ``--logprob-thold``，plan 收紧到 -0.8（默认）。
        word_timestamps: 是否启用 word-level 时间戳。仅在非 CJK 语言下追加
            ``--max-len 1 --split-on-word``，CJK 强制单字符无意义。
        max_context_zero: 是否传 ``--max-context 0``（plan 默认 True，抑制
            前文 hallucination）。
        threads: ``--threads``，None 表示让 cli 自决。
        extra: 透传给 cli 的额外 flag，追加在 argv 末尾。
    """

    model_path: Path
    language: str
    vad: bool
    vad_model_path: Path | None
    logprob_thold: float = -0.8
    word_timestamps: bool = True
    max_context_zero: bool = True
    threads: int | None = None
    extra: list[str] = field(default_factory=list)


def _strip_json_suffix(out_json: Path) -> str:
    """``-of`` 接收 *prefix*；whisper-cli 自己加 .json。"""
    s = str(out_json)
    if s.endswith(".json"):
        return s[: -len(".json")]
    return s


def build_argv(
    flags: WhisperFlags,
    audio: Path,
    out_json: Path,
    *,
    whisper_bin: Path,
) -> list[str]:
    """构造 ``whisper-cli`` argv。

    纯函数；不 spawn 也不读取磁盘。

    Mandatory:
      - ``-m <model>``
      - ``-f <audio>``
      - ``-l <language>``
      - ``-ojf``  （等价于 ``--output-json-full``）
      - ``-of <prefix>``  （whisper-cli 自动追加 .json，所以这里必须是 prefix）
      - ``--print-progress``（让 stderr 吐 progress=N% 行）
      - ``--no-prints``（屏蔽无结构 prelude，让 stderr 干净）

    Conditional:
      - ``--max-context 0``：``flags.max_context_zero`` 为真（默认）。
      - ``--logprob-thold {N}``：始终（默认 -0.8）。
      - ``--max-len 1 --split-on-word``：``word_timestamps`` 且语言**不在** CJK。
      - ``--vad --vad-model <path>``：``vad=True`` 且 ``vad_model_path`` 非 None。
      - ``--threads {N}``：仅当 ``threads is not None``。

    最后追加 ``flags.extra``。
    """
    argv: list[str] = [
        str(whisper_bin),
        "-m", str(flags.model_path),
        "-f", str(audio),
        "-l", flags.language,
        "-ojf",
        "-of", _strip_json_suffix(out_json),
        "--print-progress",
        "--no-prints",
    ]

    if flags.max_context_zero:
        argv += ["--max-context", "0"]

    # 始终带 logprob-thold（plan 默认 -0.8）
    argv += ["--logprob-thold", str(flags.logprob_thold)]

    if flags.word_timestamps and flags.language not in CJK_LANGUAGES:
        argv += ["--max-len", "1", "--split-on-word"]

    if flags.vad and flags.vad_model_path is not None:
        argv += ["--vad", "--vad-model", str(flags.vad_model_path)]

    if flags.threads is not None:
        argv += ["--threads", str(flags.threads)]

    if flags.extra:
        argv += list(flags.extra)

    return argv


# ── JSON parsing ───────────────────────────────────────────────────────
def _is_meta_token(text: str) -> bool:
    """whisper internal token 形如 ``[_BEG_]`` / ``[_TT_5]`` 偶尔会泄漏到
    transcription 列表，需要过滤。"""
    s = text.strip()
    return s.startswith("[_") and s.endswith("]")


def parse_whisper_json(raw: dict) -> list[Entry]:
    """解析 whisper.cpp ``--output-json-full`` 中的 ``transcription[]``。

    每行映射：
      - ``text`` → ``Entry.text``（保留原始前导空格）
      - ``offsets.from`` → ``t_from_ms``
      - ``offsets.to`` → ``t_to_ms``
      - ``no_speech_prob`` → 透传（缺失则 None）
      - ``raw`` → 原始 dict（debug 用）

    过滤：
      - text strip 后为空
      - meta token（``[_BEG_]`` / ``[_TT_5]`` 等）

    Returns:
        Entry 列表，保留源顺序。
    """
    entries: list[Entry] = []
    for row in raw.get("transcription", []) or []:
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        if _is_meta_token(text):
            continue
        offsets = row.get("offsets") or {}
        t_from = int(offsets.get("from", 0))
        t_to = int(offsets.get("to", 0))
        no_speech = row.get("no_speech_prob")
        if no_speech is not None:
            try:
                no_speech = float(no_speech)
            except (TypeError, ValueError):
                no_speech = None
        entries.append(
            Entry(
                text=text,
                t_from_ms=t_from,
                t_to_ms=t_to,
                no_speech_prob=no_speech,
                raw=row,
            )
        )
    return entries


# ── Run ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class WhisperRunResult:
    """``run_whisper`` 的返回值。

    Attributes:
        raw_json: 解析后的 ``--output-json-full`` 顶层 dict。
        entries: 来自 ``transcription[]`` 的 Entry 列表（已过滤空 / meta）。
        elapsed_secs: 子进程从 spawn 到 exit 的 monotonic 耗时。
    """

    raw_json: dict
    entries: list[Entry]
    elapsed_secs: float


def _resolve_json_path(out_json: Path) -> Path:
    """build_argv 传的是 prefix；whisper-cli 写到 ``{prefix}.json``。

    本函数返回最终真实路径，无论 ``out_json`` 给的是带 .json 还是不带。
    """
    s = str(out_json)
    if s.endswith(".json"):
        return Path(s)
    return Path(s + ".json")


def run_whisper(
    audio: Path,
    out_json: Path,
    flags: WhisperFlags,
    *,
    whisper_bin: Path,
    timeout_secs: float,
    env: dict[str, str] | None = None,
    progress_cb: Callable[[int], None] | None = None,
) -> WhisperRunResult:
    """拉起 ``whisper-cli`` 并阻塞等待。

    - argv 由 ``build_argv`` 构造。
    - stdout：丢弃（结果通过 ``-ojf`` 落到 JSON 文件）。
    - stderr：逐行读，正则匹配 ``PROGRESS_RE``；命中且百分比变化时调用
      ``progress_cb(percent)``。同时累加完整 stderr 用于错误回报。
    - 超时：超过 ``timeout_secs`` → kill 并抛 ``WhisperTimeout``。
    - 非零退出：抛 ``WhisperFailed``，附 returncode + stderr 末尾 50 行。
    - 成功：读 ``{prefix}.json`` 解析 → ``WhisperRunResult``。

    Args:
        audio: 输入 wav/mp3 等路径。
        out_json: 输出 JSON 路径（可带可不带 ``.json`` 后缀；统一处理）。
        flags: ``WhisperFlags`` 配置。
        whisper_bin: whisper-cli 可执行路径。
        timeout_secs: 总挂钟超时（秒）。
        env: 子进程环境；None 表示继承当前进程。
        progress_cb: 每次进度变化时回调（参数为 0-100 整数）。

    Raises:
        WhisperFailed: 非零退出。
        WhisperTimeout: 超时。
    """
    argv = build_argv(flags, audio, out_json, whisper_bin=whisper_bin)
    json_path = _resolve_json_path(out_json)

    started = time.monotonic()
    deadline = started + timeout_secs

    # 二进制 stderr：bufsize=0 让 readline 在每次换行时立刻返回，避免大块缓冲
    # 阻塞主循环超时检测。
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=env,
    )

    stderr_lines: list[bytes] = []
    last_emitted: int | None = None

    try:
        assert proc.stderr is not None
        # 后台线程读 stderr（readline 阻塞），主线程轮询 deadline / proc.poll()。
        stop_event = threading.Event()

        def _drain_stderr() -> None:
            nonlocal last_emitted
            try:
                for raw_line in iter(proc.stderr.readline, b""):
                    if stop_event.is_set():
                        break
                    if not raw_line:
                        break
                    stderr_lines.append(raw_line)
                    m = PROGRESS_RE.search(raw_line)
                    if m and progress_cb is not None:
                        try:
                            pct = int(m.group(1))
                        except ValueError:
                            continue
                        if pct != last_emitted:
                            last_emitted = pct
                            try:
                                progress_cb(pct)
                            except Exception:
                                # progress callback 异常不应中断转写
                                pass
            except Exception:
                # 读取异常（例如 stream 已关闭）静默退出
                pass

        reader = threading.Thread(target=_drain_stderr, daemon=True)
        reader.start()

        # 主循环：检查进程是否结束 / 是否超时
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            if time.monotonic() >= deadline:
                # 超时：kill + 等 reader 收尾
                stop_event.set()
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                reader.join(timeout=2)
                raise WhisperTimeout(
                    f"whisper-cli exceeded timeout of {timeout_secs}s"
                )
            time.sleep(0.05)

        # 确保 stderr 被读完
        reader.join(timeout=5)

        elapsed = time.monotonic() - started
        returncode = proc.returncode

        if returncode != 0:
            tail = b"".join(stderr_lines[-50:]).decode("utf-8", errors="replace")
            raise WhisperFailed(returncode=returncode, stderr_tail=tail)

        # 读 JSON
        try:
            with json_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError as e:
            tail = b"".join(stderr_lines[-50:]).decode("utf-8", errors="replace")
            raise WhisperFailed(
                returncode=returncode,
                stderr_tail=(
                    f"output JSON not found at {json_path}\n--- stderr tail ---\n{tail}"
                ),
            ) from e

        entries = parse_whisper_json(raw)
        return WhisperRunResult(
            raw_json=raw,
            entries=entries,
            elapsed_secs=elapsed,
        )

    finally:
        # 兜底：进程仍活着 → 杀掉
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass


__all__ = [
    "CJK_LANGUAGES",
    "PROGRESS_RE",
    "WhisperFailed",
    "WhisperTimeout",
    "WhisperFlags",
    "WhisperRunResult",
    "find_whisper_cli",
    "find_whisper_model",
    "find_vad_model",
    "build_argv",
    "parse_whisper_json",
    "run_whisper",
]
