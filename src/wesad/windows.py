"""Windowing of the prepared WESAD files (output of scripts/prepare_wesad.py).

A window is a time interval [start, end) that lies entirely inside ONE contiguous
run of a single study condition. Labels come from the RespiBAN label stream,
not from the times in SX_quest.csv (see the data audit for why).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "wesad_small"
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]  # 15 subjects, S12 does not exist
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}

# Task definitions (Schmidt et al. 2018): meditation and labels 0, 5-7 are not used.
TASKS = {
    "binary": {1: 0, 3: 0, 2: 1},          # non-stress (baseline + amusement) vs stress
    "three_class": {1: 0, 2: 1, 3: 2},     # baseline / stress / amusement
}
TASK_CLASS_NAMES = {
    "binary": ["non-stress", "stress"],
    "three_class": ["baseline", "stress", "amusement"],
}

LABEL_KEY = "chest_ECG_label"   # 350 Hz label stream: finest resolution kept in the npz
MIN_RUN_S = 30.0                # ignore label blips shorter than this (none exist in WESAD)


def load_subject(subject: str, data_dir: pathlib.Path = DATA_DIR):
    return np.load(data_dir / f"{subject}.npz")


def condition_runs(labels: np.ndarray, fs: float, subject: str) -> pd.DataFrame:
    """Contiguous runs of labels 1-4 (one row per run), in protocol order."""
    labels = np.asarray(labels).astype(np.int16)
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    rows = []
    for s, e in zip(starts, ends):
        lab = int(labels[s])
        if lab in LABEL_NAMES and (e - s) / fs >= MIN_RUN_S:
            rows.append(dict(subject=subject, label=lab, start_s=s / fs, end_s=e / fs))
    runs = pd.DataFrame(rows)
    runs["run_pos"] = np.arange(1, len(runs) + 1)  # 1 = first condition of the protocol
    runs["run_id"] = runs.subject + "_r" + runs.run_pos.astype(str)
    return runs


def windows_from_runs(runs: pd.DataFrame, win_s: float, step_s: float) -> pd.DataFrame:
    """All windows of length win_s, shifted by step_s, that fit entirely inside a run."""
    out = []
    for r in runs.itertuples(index=False):
        n = int(np.floor((r.end_s - r.start_s - win_s) / step_s + 1e-9)) + 1
        if n <= 0:
            continue
        w0 = r.start_s + step_s * np.arange(n)
        out.append(pd.DataFrame(dict(
            subject=r.subject, label=r.label, run_id=r.run_id, run_pos=r.run_pos,
            start_s=w0, end_s=w0 + win_s,
            t_mid_min=(w0 + win_s / 2) / 60.0,          # time since recording start (confound probe)
            t_in_run_s=w0 - r.start_s,
        )))
    w = pd.concat(out, ignore_index=True)
    w["win_s"], w["step_s"] = win_s, step_s
    return w


def build_window_index(win_s: float = 60.0, step_s: float = 5.0,
                       subjects=SUBJECTS, data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    parts = []
    for s in subjects:
        d = load_subject(s, data_dir)
        runs = condition_runs(d[LABEL_KEY], float(d["chest_ECG_fs"]), s)
        parts.append(windows_from_runs(runs, win_s, step_s))
    return pd.concat(parts, ignore_index=True)


def task_view(windows: pd.DataFrame, task: str) -> pd.DataFrame:
    """Keep only the windows used by `task` and add the integer target y."""
    mapping = TASKS[task]
    w = windows[windows.label.isin(mapping.keys())].copy()
    w["y"] = w.label.map(mapping).astype(int)
    return w.reset_index(drop=True)
