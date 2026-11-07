"""Per-subject physiological processing and per-window 'level' features.

Processing is done once per subject on the whole recording (peak detection, EDA
decomposition, filtering). It is unsupervised and uses no label and no other subject,
so it cannot leak labels across subjects. Filters are zero-phase (offline analysis);
this is noted as a limitation for real-time use.

Outputs per subject (data/derived/SX_physio.npz):
  event series : R-peak times, pulse-peak times + pulse amplitudes, breath peaks + amplitudes,
                 SCR peak times + amplitudes (chest, wrist)
  continuous   : EDA tonic/phasic, temperature, EMG envelope
  4 Hz grid    : node series for the coupling graphs (G1)
"""
from __future__ import annotations

import pathlib
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from .windows import DATA_DIR, load_subject

warnings.filterwarnings("ignore", module="neurokit2")
DERIVED = DATA_DIR.parent / "derived"
GRID_FS = 4.0  # common rate of the node series used by the coupling graphs

CHEST_NODES = ["hr_ecg", "resp", "eda_ph_chest", "emg_env", "act_chest"]
WRIST_NODES = ["hr_bvp", "pulse_amp", "eda_ph_wrist", "act_wrist"]


# ----------------------------------------------------------------------- helpers
def _lowpass(x, fc, fs, order=2):
    return sosfiltfilt(butter(order, fc, btype="lowpass", fs=fs, output="sos"), x)


def _highpass(x, fc, fs, order=2):
    return sosfiltfilt(butter(order, fc, btype="highpass", fs=fs, output="sos"), x)


def _to_grid(t_src, v_src, t_grid):
    """Linear interpolation of an irregular (event) series onto the grid."""
    ok = np.isfinite(v_src)
    if ok.sum() < 2:
        return np.full_like(t_grid, np.nan)
    return np.interp(t_grid, t_src[ok], v_src[ok])


def _uniform_to_grid(x, fs, t_grid):
    return np.interp(t_grid, np.arange(len(x)) / fs, x)


def _activity(acc, fs, t_grid):
    """Movement intensity: 1-s moving SD of |acc|, sampled on the grid."""
    mag = np.linalg.norm(acc, axis=1)
    k = max(1, int(round(fs)))
    s = pd.Series(mag).rolling(k, center=True, min_periods=1).std().fillna(0).values
    return _uniform_to_grid(s, fs, t_grid)


def _valid_intervals(t):
    """Inter-beat intervals with a simple artefact rule: 0.3-2.0 s and <30 % jump."""
    ibi = np.diff(t)
    ok = (ibi > 0.3) & (ibi < 2.0)
    jump = np.r_[False, np.abs(np.diff(ibi)) / ibi[:-1] > 0.3]
    return ibi, ok & ~jump


# --------------------------------------------------------------- per-subject
def process_subject(subject: str, force: bool = False) -> dict:
    DERIVED.mkdir(parents=True, exist_ok=True)
    path = DERIVED / f"{subject}_physio.npz"
    if path.exists() and not force:
        return dict(np.load(path))
    d = load_subject(subject)
    out = {}
    T = len(d["chest_ECG"]) / float(d["chest_ECG_fs"])
    tg = np.arange(0, T, 1 / GRID_FS)
    out["t_grid"] = tg

    # ECG -> R peaks
    fs = float(d["chest_ECG_fs"])
    ecg = nk.ecg_clean(d["chest_ECG"].ravel(), sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method="neurokit", correct_artifacts=True)
    r_t = np.asarray(info["ECG_R_Peaks"]) / fs
    out["r_t"] = r_t
    ibi, ok = _valid_intervals(r_t)
    hr = np.where(ok, 60.0 / ibi, np.nan)
    out["hr_ecg"] = _to_grid(r_t[1:], hr, tg)

    # BVP -> pulse peaks and pulse amplitude (peak minus preceding trough)
    fs = float(d["wrist_BVP_fs"])
    # The E4 BVP is detected on the inverted signal: on 6 subjects x 3 conditions this gave the
    # most consistent pulse timing relative to the ECG R peaks (M2 check, see report).
    bvp = -nk.ppg_clean(d["wrist_BVP"].ravel(), sampling_rate=fs)
    _, info = nk.ppg_peaks(bvp, sampling_rate=fs, method="elgendi", correct_artifacts=True)
    p = np.asarray(info["PPG_Peaks"])
    trough = np.array([bvp[max(0, a):b].min() if b > a else np.nan
                       for a, b in zip(np.r_[0, p[:-1]], p)])
    amp = bvp[p] - trough
    p_t = p / fs
    out["bvp_t"], out["bvp_amp"] = p_t, amp
    ibi, ok = _valid_intervals(p_t)
    out["hr_bvp"] = _to_grid(p_t[1:], np.where(ok, 60.0 / ibi, np.nan), tg)
    amp_ok = np.where((amp > 0) & (amp < np.nanpercentile(amp, 99.5)), amp, np.nan)
    out["pulse_amp"] = _to_grid(p_t, amp_ok, tg)

    # Respiration -> breaths (peak = end of inspiration), depth = peak minus preceding trough
    fs = float(d["chest_Resp_fs"])
    rsp = nk.rsp_clean(d["chest_Resp"].ravel(), sampling_rate=fs, method="khodadad2018")
    _, info = nk.rsp_peaks(rsp, sampling_rate=fs)
    pk, tr = np.asarray(info["RSP_Peaks"]), np.asarray(info["RSP_Troughs"])
    depth = np.array([rsp[k] - rsp[tr[tr < k][-1]] if (tr < k).any() else np.nan for k in pk])
    out["rsp_t"], out["rsp_depth"] = pk / fs, depth
    out["resp"] = _uniform_to_grid(rsp, fs, tg)

    # EDA (chest 35 Hz, wrist 4 Hz): tonic / phasic split (0.05 Hz high-pass) and SCR peaks
    for dev in ("chest", "wrist"):
        fs = float(d[f"{dev}_EDA_fs"])
        x = d[f"{dev}_EDA"].ravel().astype(np.float64)
        x = _lowpass(x, 1.0, fs) if fs > 4 else x
        tonic = _lowpass(x, 0.05, fs)
        phasic = x - tonic
        _, info = nk.eda_peaks(phasic, sampling_rate=fs, method="neurokit", amplitude_min=0.1)
        pk = np.asarray(info["SCR_Peaks"], dtype=int)
        out[f"eda_tonic_{dev}"], out[f"eda_phasic_{dev}"] = tonic, phasic
        out[f"eda_fs_{dev}"] = fs
        out[f"scr_t_{dev}"] = pk / fs
        out[f"scr_amp_{dev}"] = np.asarray(info["SCR_Amplitude"], dtype=float)
        out[f"eda_ph_{dev}"] = _uniform_to_grid(phasic, fs, tg)

    # Temperature
    out["temp_chest"], out["temp_chest_fs"] = d["chest_Temp"].ravel(), float(d["chest_Temp_fs"])
    out["temp_wrist"], out["temp_wrist_fs"] = d["wrist_TEMP"].ravel(), float(d["wrist_TEMP_fs"])

    # EMG envelope (20 Hz high-pass, rectified, 1 Hz low-pass)
    fs = float(d["chest_EMG_fs"])
    env = _lowpass(np.abs(_highpass(d["chest_EMG"].ravel().astype(np.float64), 20.0, fs)), 1.0, fs)
    out["emg_env_full"], out["emg_fs"] = env[::35], fs / 35  # keep 10 Hz copy
    out["emg_env"] = _uniform_to_grid(env, fs, tg)

    # Activity (movement) nodes
    out["act_chest"] = _activity(d["chest_ACC"].astype(np.float64), float(d["chest_ACC_fs"]), tg)
    out["act_wrist"] = _activity(d["wrist_ACC"].astype(np.float64) / 64.0, float(d["wrist_ACC_fs"]), tg)

    np.savez_compressed(path, **out)
    return out


# --------------------------------------------------------------- window features
def _slice_events(t, a, b):
    i, j = np.searchsorted(t, a), np.searchsorted(t, b)
    return i, j


def _hrv(beat_t, a, b, prefix, freq=True):
    i, j = _slice_events(beat_t, a, b)
    t = beat_t[i:j]
    f = {}
    if len(t) < 5:
        return {f"{prefix}_{k}": np.nan for k in ("hr_mean", "hr_std", "sdnn", "rmssd", "pnn50", "lf", "hf", "lf_hf", "valid")}
    ibi, ok = _valid_intervals(t)
    v = ibi[ok]
    pair = ok[1:] & ok[:-1]
    d = np.diff(ibi)[pair]
    f[f"{prefix}_valid"] = ok.mean()
    f[f"{prefix}_hr_mean"] = 60.0 / v.mean() if len(v) else np.nan
    f[f"{prefix}_hr_std"] = np.std(60.0 / v) if len(v) > 2 else np.nan
    f[f"{prefix}_sdnn"] = v.std() * 1000 if len(v) > 2 else np.nan
    f[f"{prefix}_rmssd"] = np.sqrt(np.mean(d ** 2)) * 1000 if len(d) > 2 else np.nan
    f[f"{prefix}_pnn50"] = np.mean(np.abs(d) > 0.05) if len(d) > 2 else np.nan
    lf = hf = np.nan
    if freq and len(v) > 10:
        tg = np.arange(t[1], t[-1], 0.25)
        x = np.interp(tg, t[1:][ok], ibi[ok])
        x = x - np.polyval(np.polyfit(tg, x, 1), tg)
        fr, p = welch(x, fs=4.0, nperseg=min(len(x), 256))
        lf = np.trapezoid(p[(fr >= 0.04) & (fr < 0.15)], fr[(fr >= 0.04) & (fr < 0.15)])
        hf = np.trapezoid(p[(fr >= 0.15) & (fr < 0.40)], fr[(fr >= 0.15) & (fr < 0.40)])
    f[f"{prefix}_lf"] = np.log(lf * 1e6 + 1e-6) if np.isfinite(lf) else np.nan
    f[f"{prefix}_hf"] = np.log(hf * 1e6 + 1e-6) if np.isfinite(hf) else np.nan
    f[f"{prefix}_lf_hf"] = (np.log(lf / hf) if (np.isfinite(lf) and hf > 0 and lf > 0) else np.nan)
    return f


def _slope_per_min(x, fs):
    if len(x) < 3:
        return np.nan
    t = np.arange(len(x)) / fs / 60.0
    return np.polyfit(t, x, 1)[0]


def window_level_features(P: dict, a: float, b: float) -> dict:
    f = {}
    f.update(_hrv(P["r_t"], a, b, "ecg", freq=(b - a) >= 30))
    f.update(_hrv(P["bvp_t"], a, b, "bvp", freq=(b - a) >= 30))
    i, j = _slice_events(P["bvp_t"], a, b)
    amp = P["bvp_amp"][i:j]
    f["bvp_amp_mean"] = np.nanmedian(amp) if len(amp) else np.nan
    f["bvp_amp_cv"] = np.nanstd(amp) / np.nanmean(amp) if len(amp) > 2 else np.nan

    # respiration
    i, j = _slice_events(P["rsp_t"], a, b)
    bt, dep = P["rsp_t"][i:j], P["rsp_depth"][i:j]
    bi = np.diff(bt)
    f["resp_rate"] = 60.0 / np.median(bi) if len(bi) >= 2 else np.nan
    f["resp_int_cv"] = bi.std() / bi.mean() if len(bi) >= 3 else np.nan
    f["resp_depth_mean"] = np.nanmean(dep) if len(dep) else np.nan
    f["resp_depth_cv"] = np.nanstd(dep) / np.nanmean(dep) if len(dep) > 2 else np.nan

    # EDA
    for dev in ("chest", "wrist"):
        fs = float(P[f"eda_fs_{dev}"])
        ia, ib = int(round(a * fs)), int(round(b * fs))
        ton, ph = P[f"eda_tonic_{dev}"][ia:ib], P[f"eda_phasic_{dev}"][ia:ib]
        i, j = _slice_events(P[f"scr_t_{dev}"], a, b)
        amps = P[f"scr_amp_{dev}"][i:j]
        f[f"eda_{dev}_scl_mean"] = ton.mean()
        f[f"eda_{dev}_scl_slope"] = _slope_per_min(ton, fs)
        f[f"eda_{dev}_phasic_std"] = ph.std()
        f[f"eda_{dev}_scr_rate"] = (j - i) / ((b - a) / 60.0)
        f[f"eda_{dev}_scr_amp"] = np.nanmean(amps) if len(amps) else 0.0

    # temperature
    for dev in ("chest", "wrist"):
        fs = float(P[f"temp_{dev}_fs"])
        x = P[f"temp_{dev}"][int(round(a * fs)):int(round(b * fs))]
        f[f"temp_{dev}_mean"] = x.mean()
        f[f"temp_{dev}_slope"] = _slope_per_min(x, fs)

    # EMG
    fs = float(P["emg_fs"])
    e = P["emg_env_full"][int(round(a * fs)):int(round(b * fs))]
    f["emg_env_mean"] = np.log(e.mean() + 1e-9)
    f["emg_env_cv"] = e.std() / (e.mean() + 1e-12)
    return f


def node_series(P: dict, a: float, b: float, nodes) -> np.ndarray:
    """(n_nodes, n_samples) node series on the 4 Hz grid for one window."""
    ia, ib = int(round(a * GRID_FS)), int(round(b * GRID_FS))
    return np.vstack([P[n][ia:ib] for n in nodes])


# feature groups (column prefixes) used by the experiments
GROUPS = {
    "chest_levels": ("ecg_", "resp_", "eda_chest_", "temp_chest_", "emg_"),
    "wrist_levels": ("bvp_", "eda_wrist_", "temp_wrist_"),
}


def group_columns(columns, group):
    pref = GROUPS[group]
    return [c for c in columns if c.startswith(pref) and not c.endswith("_valid")]
