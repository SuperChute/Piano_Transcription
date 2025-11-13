#!/usr/bin/env python3
"""
onset_probe.py — Minimal script to test onset detection on an input audio file.

What it does
------------
- Loads an audio file (mono).
- Computes an onset envelope and detects onset times.
- Saves a CSV of onset times.
- (Optional) Saves a "click track" WAV with clicks at detected onsets, and/or a plot PNG.

NOTES:
- No pitch/FFT/note detection logic is implemented here by us (librosa handles features internally).
- Keep this focused on onsets only.

Usage
-----
python onset_probe.py input.wav \
  --sr 22050 \
  --hop-length 512 \
  --backtrack \
  --delta 0.0 \
  --save-clicks clicks.wav \
  --save-plot onsets.png \
  --save-csv onsets.csv

Dependencies
------------
pip install librosa soundfile matplotlib numpy
"""

import argparse
import csv
import os
from typing import Optional

import numpy as np

import librosa
import librosa.display  # noqa: F401 (only used when plotting)
import soundfile as sf

# -----------------------------
# Helpers
# -----------------------------

def detect_onsets(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    backtrack: bool = False,
    delta: float = 0.0,
    wait: int = 1,
    pre_max: int = 20,
    post_max: int = 20,
    pre_avg: int = 100,
    post_avg: int = 100,
    aggregate: str = "mean",
):
    """
    Run onset detection and return (onset_frames, onset_times, onset_envelope).

    Parameters mirror librosa.onset.onset_detect and onset_strength where useful.
    """
    agg_fn = np.mean if aggregate == "mean" else np.median
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=agg_fn
    )

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack,
        delta=delta,
        wait=wait,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        units="frames",
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_frames, onset_times, onset_env


def save_csv(onset_times: np.ndarray, path: str):
    """Save onset times (seconds) to CSV with columns: index,time_seconds"""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "time_seconds"])
        for i, t in enumerate(onset_times):
            writer.writerow([i, f"{t:.6f}"])


def make_click_track(
    onset_times: np.ndarray,
    sr: int,
    length_samples: Optional[int] = None,
    click_freq: float = 1000.0,
    click_duration: float = 0.03,
    click_amp: float = 0.9,
):
    """
    Create a click track signal (mono) with short tones at each onset time.
    Uses librosa.clicks for timing envelope, then replaces with a short sine "pip".
    """
    if length_samples is None:
        if onset_times.size == 0:
            length_samples = int(sr * 1.0)
        else:
            length_samples = int(sr * (onset_times.max() + 1.0))

    # Start with silence
    y_clicks = np.zeros(length_samples, dtype=np.float32)

    # Generate a single click (sine pip)
    n_click = int(sr * click_duration)
    n_click = max(1, n_click)
    t = np.arange(n_click) / sr
    single_click = (click_amp * np.sin(2 * np.pi * click_freq * t)).astype(np.float32)

    for t_sec in onset_times:
        idx = int(t_sec * sr)
        if idx < length_samples:
            end = min(length_samples, idx + n_click)
            span = end - idx
            y_clicks[idx:end] += single_click[:span]

    # Normalize to avoid clipping if overlaps
    peak = np.max(np.abs(y_clicks)) or 1.0
    y_clicks = (y_clicks / peak * 0.95).astype(np.float32)
    return y_clicks


def plot_results(
    y: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    onset_env: np.ndarray,
    hop_length: int,
    out_path: str,
):
    import matplotlib.pyplot as plt

    # Waveform with onsets
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    for t in onset_times:
        ax1.axvline(t, linestyle="--")
    ax1.set_title("Waveform with detected onsets")
    ax1.set_xlabel("Time (s)")

    # Onset envelope (separate figure to keep it clean as per good plotting practice)
    plt.figure(figsize=(12, 3))
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    plt.plot(times, onset_env)
    for t in onset_times:
        plt.axvline(t, linestyle="--")
    plt.title("Onset strength envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Strength")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close("all")


# -----------------------------
# Main entry
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Test onset detection on an audio file.")
    parser.add_argument("input_file", help="Path to input audio (e.g., hot_cross_buns.wav)")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate for loading audio (use 0 to keep native)")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for analysis")
    parser.add_argument("--backtrack", action="store_true", help="Backtrack onsets to nearest preceding minimum")
    parser.add_argument("--delta", type=float, default=0.0, help="Onset threshold offset")
    parser.add_argument("--wait", type=int, default=1, help="Minimum frames between onsets")
    parser.add_argument("--pre-max", type=int, default=20, help="Before-maximum window (frames)")
    parser.add_argument("--post-max", type=int, default=20, help="After-maximum window (frames)")
    parser.add_argument("--pre-avg", type=int, default=100, help="Before-average window (frames)")
    parser.add_argument("--post-avg", type=int, default=100, help="After-average window (frames)")
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean", help="Aggregation for onset envelope")
    parser.add_argument("--save-csv", default=None, help="Where to save onset times CSV")
    parser.add_argument("--save-clicks", default=None, help="Where to save click track WAV")
    parser.add_argument("--save-plot", default=None, help="Where to save plot PNG")

    args = parser.parse_args()

    # Load audio
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    target_sr = None if args.sr == 0 else args.sr
    y, sr = librosa.load(args.input_file, sr=target_sr, mono=True)


    # Detect onsets
    frames, times, onset_env = detect_onsets(
        y=y,
        sr=sr,
        hop_length=args.hop_length,
        backtrack=args.backtrack,
        delta=args.delta,
        wait=args.wait,
        pre_max=args.pre_max,
        post_max=args.post_max,
        pre_avg=args.pre_avg,
        post_avg=args.post_avg,
        aggregate=args.aggregate,
    )

    # Report to console
    print(f"Detected {len(times)} onsets (seconds):")
    print(" ".join(f"{t:.3f}" for t in times))

    # Save CSV
    if args.save_csv:
        save_csv(times, args.save_csv)
        print(f"[Saved] CSV -> {args.save_csv}")

    # Save click track
    if args.save_clicks:
        y_clicks = make_click_track(times, sr=sr, length_samples=len(y))
        sf.write(args.save_clicks, y_clicks, sr)
        print(f"[Saved] Clicks WAV -> {args.save_clicks}")

    # Save plot
    if args.save_plot:
        plot_results(y, sr, times, onset_env, args.hop_length, args.save_plot)
        print(f"[Saved] Plot -> {args.save_plot}")


if __name__ == "__main__":
    main()
