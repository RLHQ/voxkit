"""Unit tests for voxkit.core.subtitle_metrics."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from voxkit.core.semantic_resegment import ResegmentParams, SubtitleCue
from voxkit.core.subtitle_metrics import SubtitleMetrics, compute_subtitle_metrics


def _cue(start: float, end: float, text: str = "hello") -> SubtitleCue:
    return SubtitleCue(start=start, end=end, speaker="Speaker 1", text=text)


def test_empty_cues_return_zero_metrics():
    metrics = compute_subtitle_metrics([], ResegmentParams())

    assert metrics == SubtitleMetrics(
        cueCount=0,
        avgCueDurS=0.0,
        p50CueDurS=0.0,
        p90CueDurS=0.0,
        flashCueRate=0.0,
        longCueRate=0.0,
        avgChars=0.0,
        overCharLimitRate=0.0,
        overCpsRate=0.0,
    )
    assert metrics.to_dict() == asdict(metrics)


def test_normal_cues_compute_counts_averages_and_percentiles():
    params = ResegmentParams(min_dur_s=1.5, max_dur_s=7.0, max_chars=20, max_cps=22.0)
    cues = [
        _cue(0.0, 2.0, "abcd"),
        _cue(2.0, 5.0, "abcdef"),
        _cue(5.0, 9.0, "abcdefgh"),
    ]

    metrics = compute_subtitle_metrics(cues, params)

    assert metrics.cueCount == 3
    assert metrics.avgCueDurS == pytest.approx(3.0)
    assert metrics.p50CueDurS == pytest.approx(3.0)
    assert metrics.p90CueDurS == pytest.approx(3.8)
    assert metrics.flashCueRate == 0.0
    assert metrics.longCueRate == 0.0
    assert metrics.avgChars == pytest.approx(6.0)
    assert metrics.overCharLimitRate == 0.0
    assert metrics.overCpsRate == 0.0


def test_short_cue_increments_flash_rate():
    params = ResegmentParams(min_dur_s=1.5)
    cues = [
        _cue(0.0, 1.0),
        _cue(1.0, 3.0),
    ]

    metrics = compute_subtitle_metrics(cues, params)

    assert metrics.flashCueRate == pytest.approx(0.5)


def test_long_cue_increments_long_rate():
    params = ResegmentParams(max_dur_s=7.0)
    cues = [
        _cue(0.0, 8.0),
        _cue(8.0, 12.0),
    ]

    metrics = compute_subtitle_metrics(cues, params)

    assert metrics.longCueRate == pytest.approx(0.5)


def test_over_character_limit_rate_uses_params_max_chars():
    params = ResegmentParams(max_chars=5)
    cues = [
        _cue(0.0, 2.0, "12345"),
        _cue(2.0, 4.0, "123456"),
    ]

    metrics = compute_subtitle_metrics(cues, params)

    assert metrics.avgChars == pytest.approx(5.5)
    assert metrics.overCharLimitRate == pytest.approx(0.5)


def test_over_cps_rate_uses_params_max_cps():
    params = ResegmentParams(max_cps=4.0)
    cues = [
        _cue(0.0, 2.0, "12345678"),
        _cue(2.0, 4.0, "123456789"),
    ]

    metrics = compute_subtitle_metrics(cues, params)

    assert metrics.overCpsRate == pytest.approx(0.5)


def test_zero_and_reversed_duration_boundaries_do_not_raise():
    params = ResegmentParams(min_dur_s=1.5, max_cps=22.0)
    cues = [
        _cue(2.0, 2.0, "instant"),
        _cue(5.0, 4.0, "reversed"),
        _cue(7.0, 7.0, ""),
    ]

    metrics = compute_subtitle_metrics(cues, params)

    assert metrics.cueCount == 3
    assert metrics.avgCueDurS == 0.0
    assert metrics.p50CueDurS == 0.0
    assert metrics.p90CueDurS == 0.0
    assert metrics.flashCueRate == 1.0
    assert metrics.longCueRate == 0.0
    assert metrics.overCpsRate == pytest.approx(2 / 3)
