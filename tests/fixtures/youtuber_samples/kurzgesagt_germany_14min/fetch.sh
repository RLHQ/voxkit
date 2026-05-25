#!/usr/bin/env bash
# 按需下载 Kurzgesagt "GERMANY IS OVER" 的 16kHz mono WAV 音频。
# 字幕已入仓；音频不入仓（版权 / 体积），本脚本按需拉取。
#
# 用法：
#   ./fetch.sh           # 拉音频，落到 audio.wav
#   ./fetch.sh --force   # 已存在也重拉
#
# 依赖：yt-dlp、ffmpeg（PATH 中可见）。
set -euo pipefail

VIDEO_ID="n-gYFcVx-8Y"
URL="https://www.youtube.com/watch?v=${VIDEO_ID}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${HERE}/audio.wav"

if [[ "${1:-}" != "--force" && -f "${OUT}" ]]; then
  echo "[fetch] audio.wav 已存在，跳过下载（用 --force 重拉）"
  exit 0
fi

for bin in yt-dlp ffmpeg; do
  if ! command -v "${bin}" >/dev/null 2>&1; then
    echo "[fetch] 缺少依赖：${bin}（请先安装）" >&2
    exit 1
  fi
done

echo "[fetch] 下载 ${URL} 到 ${OUT}"
yt-dlp \
  -x --audio-format wav \
  --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
  -o "${HERE}/audio.%(ext)s" \
  "${URL}"

echo "[fetch] 完成：${OUT}"
ls -lh "${OUT}"
