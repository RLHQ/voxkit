# xnzxnz_first_look_10min

**频道**：小宁子 XNZ（@xnzxnz）
**视频**：【First Look】Steam Has Released a Controller! Why Is It So Hot?
**URL**：<https://www.youtube.com/watch?v=2_FpQj_f69g>
**时长**：606 秒（10:06）
**音频**：按需 `./fetch.sh` 拉取（约 19MB，不入仓）
**字幕**：zh-Hans / zh-Hant / en-US（入仓）
**下载日期**：2026-05-13

## 字幕轨道（YouTube 原始声明）

| 文件 | 来源轨道 | 说明 |
|---|---|---|
| `sub.zh-Hans.srt` | zh-CN / zh-Hans（同源） | 作者上传的简体中文字幕；YouTube 的 zh-CN 与 zh-Hans 轨道 byte-identical，已合并 |
| `sub.zh-Hant.srt` | zh-Hant | YouTube 从 zh-CN **机器繁体转换**而来（断句相同，仅做"硬件→硬體"这类字形替换） |
| `sub.en-US.srt` | en-US | 作者上传的英文翻译，**每条与 zh-Hans 同时间窗口对齐** |

> zh-CN 和 zh-Hans 内容完全一样，仓库内只保留 zh-Hans 一份。

## 为什么入选

- **中文 reseg 金标**：作者人工断句严格服务于口播气口，不会把财经/科技术语腰斩。
- **zh-en translate 双向对照**：zh-Hans 与 en-US 同帧对齐，是把 voxkit `translate` 输出做 BLEU/编辑距离对照的真金标。
- **简繁字形对照**：zh-Hant 是机器繁体化样本，适合验证 voxkit 是否正确保留断句结构。

## voxkit 用途映射

- `transcribe` → 用 audio.wav 跑 ASR，与 sub.zh-Hans.srt 做 WER 对照；
- `reseg` → 把 transcribe 原始 cue 喂给 reseg，与 sub.zh-Hans.srt 的人工断句对照；
- `translate` zh→en → 译文与 sub.en-US.srt 对照；
- `proofread`（中文）→ 用 sub.zh-Hans.srt 作为术语写法基准。

## 音频规格

WAV / 16000 Hz / mono / 605.6 秒。按需用同目录 `./fetch.sh` 拉取。
