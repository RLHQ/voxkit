"""NDJSON 事件协议：`--json-events` 模式下 stderr 的机器可读输出。

每行一个 JSON 对象，schema：
- progress: {event, stage, percent}
- warn:     {event, code, message}
- error:    {event, code, message, fix?}
- done:     {event, elapsed_secs}

stdout 保留给最终 JSON，方便 `jq` / TS 直接消费。
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional, TextIO


class ProgressEmitter:
    """把进度/警告/错误事件以人类可读或 NDJSON 形式输出到 stderr。"""

    def __init__(self, *, json_events: bool, stream: Optional[TextIO] = None) -> None:
        self.json_events = json_events
        self.stream = stream or sys.stderr

    # ── public api ──────────────────────────────────────────────
    def progress(self, stage: str, percent: int) -> None:
        if self.json_events:
            self._emit({"event": "progress", "stage": stage, "percent": percent})
        else:
            self._human(f"[{stage}] {percent}%")

    def warn(self, code: str, message: str) -> None:
        if self.json_events:
            self._emit({"event": "warn", "code": code, "message": message})
        else:
            self._human(f"[warn:{code}] {message}")

    def error(self, code: str, message: str, fix: Optional[str] = None) -> None:
        if self.json_events:
            payload: dict[str, Any] = {"event": "error", "code": code, "message": message}
            if fix:
                payload["fix"] = fix
            self._emit(payload)
        else:
            line = f"[error:{code}] {message}"
            if fix:
                line += f"\n  fix: {fix}"
            self._human(line)

    def done(self, elapsed_secs: float) -> None:
        if self.json_events:
            self._emit({"event": "done", "elapsed_secs": round(elapsed_secs, 3)})
        else:
            self._human(f"[done] elapsed={elapsed_secs:.2f}s")

    def info(self, message: str) -> None:
        """只在人类模式下打印；NDJSON 模式忽略，避免污染流。"""
        if not self.json_events:
            self._human(message)

    # ── internals ───────────────────────────────────────────────
    def _emit(self, payload: dict[str, Any]) -> None:
        self.stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.stream.flush()

    def _human(self, line: str) -> None:
        self.stream.write(line + "\n")
        self.stream.flush()


__all__ = ["ProgressEmitter"]
