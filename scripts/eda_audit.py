"""
eda_audit.py - exploratory audit of the raw WESAD dataset (no modelling).

Reads each SX.pkl at full resolution (one subject at a time) and writes tables
to --out:
  inventory.csv          per subject / signal / channel: n, fs, duration, NaN, range, flat, clipping
  label_segments.csv     every contiguous run of a label value (start/end/duration)
  schedule_check.csv     quest.csv schedule vs label runs (checks the min.sec format)
  condition_features.csv per subject x condition quick physiological summaries
  subject_qc.csv         per-subject quality indicators (ECG polarity, RR validity, sync lag, ...)
  questionnaires.csv     parsed self-reports, one row per subject x condition

These are *descriptive* quick checks (simple peak detectors, no tuned pipeline).
"""
import argparse
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt

FS_CHEST = 700
WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
COND = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
QUEST_NAME_TO_LABEL = {"Base": 1, "TSST": 2, "Fun": 3, "Medi 1": 4, "Medi 2": 4}


# ----------------------------------------------------------------- helpers
def true_runs(mask):
    """Start (inclusive) and end (exclusive) indices of runs of True."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def label_segments(lab):
    idx = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate([[0], idx])
    ends = np.concatenate([idx, [len(lab)]])
    return starts, ends, lab[starts]


def labels_at(lab, fs, n):
    idx = np.minimum((np.arange(n) * (FS_CHEST / fs)).astype(np.int64), len(lab) - 1)
    return lab[idx]


def flat_fraction(x, fs, min_sec=5.0):
    """Fraction of samples lying in runs where the signal is exactly constant for >= min_sec."""
    same = np.diff(x) == 0
    s, e = true_runs(same)
    L = e - s + 1  # run of equal diffs of length k means k+1 equal samples
    return float(L[L >= min_sec * fs].sum()) / len(x)


def sos_bp(lo, hi, fs, order=2):
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def detect_rpeaks(ecg, fs=FS_CHEST):
    """Very simple R-peak detector: 5-15 Hz band-pass, squared, adaptive threshold per 30 s."""
    y = sosfiltfilt(sos_bp(5, 15, fs), ecg)
    env = y ** 2
    chunk = 30 * fs
    peaks = []
    for a in range(0, len(env), chunk):
        seg = env[a:a + chunk]
        if len(seg) < fs:
            continue
        thr = 0.35 * np.percentile(seg, 99.5)
        p, _ = find_peaks(seg, height=thr, distance=int(0.3 * fs))
        peaks.append(p + a)
    return np.concatenate(peaks) if peaks else np.array([], dtype=int)


def ecg_polarity(ecg, peaks, fs=FS_CHEST):
    """Fraction of R peaks whose dominant deflection (|.| max within +-40 ms) is positive."""
    hp = sosfiltfilt(butter(2, 0.5, btype="highpass", fs=fs, output="sos"), ecg)
    w = int(0.04 * fs)
    pk = peaks[(peaks > w) & (peaks < len(hp) - w)]
    idx = pk[:, None] + np.arange(-w, w + 1)[None, :]
    win = hp[idx]
    j = np.argmax(np.abs(win), axis=1)
    return float(np.mean(win[np.arange(len(pk)), j] > 0))


def detect_bvp_peaks(bvp, fs=64):
    y = sosfiltfilt(sos_bp(0.7, 3.5, fs), bvp)
    p, _ = find_peaks(y, distance=int(0.33 * fs), prominence=0.3 * np.std(y))
    return p


def rr_stats(peaks, lab_at_peak, fs):
    """Per-condition HR (from median RR), RMSSD (ms) and fraction of plausible RRs."""
    out = {}
    t = peaks / fs
    rr = np.diff(t)
    lab_pair = np.where(lab_at_peak[1:] == lab_at_peak[:-1], lab_at_peak[1:], -1)
    for c in COND:
        r = rr[lab_pair == c]
        if len(r) < 10:
            out[c] = (np.nan, np.nan, np.nan, len(r))
            continue
        ok = (r > 0.3) & (r < 2.0)
        rv = r[ok]
        d = np.diff(r)
        okd = ok[1:] & ok[:-1] & (np.abs(d) < 0.25)
        rmssd = float(np.sqrt(np.mean(d[okd] ** 2)) * 1000) if okd.sum() > 10 else np.nan
        hr = float(60.0 / np.median(rv)) if len(rv) else np.nan
        out[c] = (hr, rmssd, float(ok.mean()), len(r))
    return out


def resp_peaks(resp700):
    x = resample_poly(resp700, 1, 20)  # 35 Hz
    fs = FS_CHEST / 20
    y = sosfiltfilt(sos_bp(0.1, 0.5, fs), x)
    p, _ = find_peaks(y, distance=int(2.0 * fs), prominence=0.5 * np.std(y))
    return p / fs  # seconds


def sync_lag(chest_acc, wrist_acc, max_lag=120):
    """Cross-correlate 1-s motion-energy series of the two devices; returns (r at lag0, best lag s, r at best)."""
    a = np.linalg.norm(chest_acc, axis=1)
    n = len(a) // FS_CHEST
    ea = a[:n * FS_CHEST].reshape(n, FS_CHEST).std(axis=1)
    b = np.linalg.norm(wrist_acc / 64.0, axis=1)
    m = len(b) // 32
    eb = b[:m * 32].reshape(m, 32).std(axis=1)
    L = min(n, m)
    ea, eb = ea[:L], eb[:L]
    ea = (ea - ea.mean()) / (ea.std() + 1e-12)
    eb = (eb - eb.mean()) / (eb.std() + 1e-12)
    best = (0, -2)
    r0 = float(np.mean(ea * eb))
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.mean(ea[lag:] * eb[:L - lag])
        else:
            r = np.mean(ea[:L + lag] * eb[-lag:])
        if r > best[1]:
            best = (lag, float(r))
    return r0, best[0], best[1]


# ------------------------------------------------------------ quest.csv parsing
def parse_quest(path):
    rows = [ln.rstrip("\n").split(";") for ln in open(path, encoding="utf-8")]
    q = {"PANAS": [], "STAI": [], "DIM": [], "SSSQ": []}
    for r in rows:
        key = r[0].strip("# ").strip()
        vals = [v for v in r[1:] if v != ""]
        if key == "ORDER":
            q["order"] = vals
        elif key == "START":
            q["start_raw"] = vals
        elif key == "END":
            q["end_raw"] = vals
        elif key in q:
            q[key].append([float(v) for v in vals])
    return q


def minsec_to_s(txt, pad_two_digits):
    if "." not in txt:  # whole minutes, e.g. "32" -> 32:00
        return int(txt) * 60
    mm, ss = txt.split(".")
    if pad_two_digits:
        ss = ss.ljust(2, "0")  # "50.3" -> 50 min 30 s (trailing zero dropped)
    return int(mm) * 60 + int(ss)


# ------------------------------------------------------------------- main
def audit_subject(pkl_path, quest_path, rows):
    d = pickle.load(open(pkl_path, "rb"), encoding="latin1")
    sid = str(d["subject"])
    lab = np.asarray(d["label"]).ravel().astype(np.int16)
    sig = d["signal"]
    print(f"{sid}: loaded", flush=True)

    # ---- inventory
    for dev in ("chest", "wrist"):
        for name, v in sig[dev].items():
            v = np.asarray(v)
            fs = FS_CHEST if dev == "chest" else WRIST_FS[name]
            x2 = v.reshape(len(v), -1)
            for ch in range(x2.shape[1]):
                x = x2[:, ch].astype(np.float64)
                finite = np.isfinite(x)
                xf = x[finite]
                lo, hi = xf.min(), xf.max()
                rows["inventory"].append(dict(
                    subject=sid, device=dev, signal=name, channel=ch, fs=fs, n=len(x),
                    duration_s=len(x) / fs, nan=int((~finite).sum()),
                    min=lo, p1=np.percentile(xf, 1), median=np.median(xf),
                    p99=np.percentile(xf, 99), max=hi, mean=xf.mean(), std=xf.std(),
                    flat_frac=flat_fraction(x, fs), clip_frac=float(np.mean((x == lo) | (x == hi)))))

    # ---- label segments and schedule check
    st, en, lv = label_segments(lab)
    for s, e, l in zip(st, en, lv):
        rows["segments"].append(dict(subject=sid, label=int(l), start_s=s / FS_CHEST,
                                     end_s=e / FS_CHEST, dur_s=(e - s) / FS_CHEST))
    q = parse_quest(quest_path)
    names = [n for n in q["order"] if not n.endswith("Read")]
    n_label4_seen = 0
    for i, nm in enumerate(q["order"]):
        if nm.endswith("Read"):
            continue
        lbl = QUEST_NAME_TO_LABEL[nm]
        runs = [(s / FS_CHEST, e / FS_CHEST) for s, e, l in zip(st, en, lv) if l == lbl and (e - s) > 30 * FS_CHEST]
        # the two meditations share label 4: the k-th meditation entry in ORDER <-> k-th label-4 run
        k = n_label4_seen if lbl == 4 else 0
        n_label4_seen += lbl == 4
        ls, le = runs[k] if k < len(runs) else (np.nan, np.nan)
        rec = dict(subject=sid, condition=nm, label=lbl, label_start_s=ls, label_end_s=le)
        for pad in (False, True):
            tag = "padded" if pad else "plain"
            rec[f"quest_start_{tag}_s"] = minsec_to_s(q["start_raw"][i], pad)
            rec[f"quest_end_{tag}_s"] = minsec_to_s(q["end_raw"][i], pad)
        rec["quest_start_raw"], rec["quest_end_raw"] = q["start_raw"][i], q["end_raw"][i]
        rows["schedule"].append(rec)

    # ---- questionnaires
    for i, nm in enumerate(names):
        stai = np.array(q["STAI"][i])
        stai_score = float(stai[[1, 2, 4]].sum() + (5 - stai[[0, 3, 5]]).sum())  # reverse at-ease/relaxed/pleasant
        panas = q["PANAS"][i]
        rows["quest"].append(dict(
            subject=sid, condition=nm, position=i + 1,
            stai_state=stai_score, sam_valence=q["DIM"][i][0], sam_arousal=q["DIM"][i][1],
            panas_stressed=panas[20], panas_nervous=panas[15], panas_items=len(panas)))

    # ---- chest/wrist handles
    ecg = np.asarray(sig["chest"]["ECG"]).ravel().astype(np.float64)
    eda_c = np.asarray(sig["chest"]["EDA"]).ravel().astype(np.float64)
    tmp_c = np.asarray(sig["chest"]["Temp"]).ravel().astype(np.float64)
    emg_c = np.asarray(sig["chest"]["EMG"]).ravel().astype(np.float64)
    acc_c = np.asarray(sig["chest"]["ACC"]).astype(np.float64)
    resp_c = np.asarray(sig["chest"]["Resp"]).ravel().astype(np.float64)
    acc_w = np.asarray(sig["wrist"]["ACC"]).astype(np.float64)
    bvp_w = np.asarray(sig["wrist"]["BVP"]).ravel().astype(np.float64)
    eda_w = np.asarray(sig["wrist"]["EDA"]).ravel().astype(np.float64)
    tmp_w = np.asarray(sig["wrist"]["TEMP"]).ravel().astype(np.float64)

    # ---- ECG
    peaks = detect_rpeaks(ecg)
    pol = ecg_polarity(ecg, peaks)
    ecg_stats = rr_stats(peaks, lab[peaks], FS_CHEST)
    # ---- BVP
    bp = detect_bvp_peaks(bvp_w)
    lab_bvp = labels_at(lab, 64, len(bvp_w))
    bvp_stats = rr_stats(bp, lab_bvp[bp], 64)
    # ---- Respiration
    rp = resp_peaks(resp_c)
    lab_rp = lab[np.minimum((rp * FS_CHEST).astype(int), len(lab) - 1)]
    # ---- motion
    acc_c_norm = np.linalg.norm(acc_c, axis=1)
    acc_w_norm = np.linalg.norm(acc_w / 64.0, axis=1)
    lab_accw = labels_at(lab, 32, len(acc_w_norm))
    lab_w4 = labels_at(lab, 4, len(eda_w))
    tmin = np.arange(len(lab)) / FS_CHEST / 60.0

    # samples where every chest channel sits on its ADC rail (found in S3): exclude from chest means
    good = np.ones(len(lab), dtype=bool)
    gs, ge = true_runs(tmp_c < -100)
    for s, e in zip(gs, ge):
        good[max(0, s - 350):e + 350] = False
    rows["qc_glitch"].append(dict(subject=sid, glitch_samples=int((~good).sum())))

    for c, cname in COND.items():
        m = (lab == c) & good
        if m.sum() == 0:
            continue
        dur = m.sum() / FS_CHEST
        mw4 = lab_w4 == c
        runs_c = true_runs(m)
        single = len(runs_c[0]) == 1
        slope_t = np.nan
        if single:
            sub = np.flatnonzero(m)[::100]
            slope_t = float(np.polyfit(tmin[sub], tmp_c[sub], 1)[0])
        hr, rmssd, rrok, nrr = ecg_stats[c]
        bhr, _, _, _ = bvp_stats[c]
        rows["cond"].append(dict(
            subject=sid, label=c, condition=cname, minutes=dur / 60, n_runs=len(runs_c[0]),
            hr_ecg=hr, rmssd_ms=rmssd, rr_valid_frac=rrok, hr_bvp=bhr,
            eda_chest=eda_c[m].mean(), eda_wrist=eda_w[mw4].mean(),
            temp_chest=tmp_c[m].mean(), temp_wrist=tmp_w[mw4].mean(),
            temp_chest_slope_c_per_min=slope_t,
            resp_rate_bpm=float((lab_rp == c).sum() / (dur / 60)),
            emg_chest_std=emg_c[m].std(), acc_chest_dyn_g=acc_c_norm[m].std(),
            acc_chest_norm_mean_g=acc_c_norm[m].mean(),
            acc_wrist_dyn_g=acc_w_norm[lab_accw == c].std(),
            acc_wrist_norm_mean_g=acc_w_norm[lab_accw == c].mean()))

    # ---- subject QC
    r0, lag, rbest = sync_lag(acc_c, acc_w)
    both = ~np.isnan(ecg_stats[1][0]) and ~np.isnan(bvp_stats[1][0])
    rows["qc"].append(dict(
        subject=sid, chest_dur_s=len(ecg) / FS_CHEST,
        wrist_bvp_dur_s=len(bvp_w) / 64, wrist_eda_dur_s=len(eda_w) / 4,
        wrist_acc_dur_s=len(acc_w) / 32, wrist_temp_dur_s=len(tmp_w) / 4,
        frac_label0=float(np.mean(lab == 0)), frac_label567=float(np.mean(lab >= 5)),
        frac_used_1to4=float(np.mean((lab >= 1) & (lab <= 4))),
        n_rpeaks=len(peaks), ecg_polarity_pos_frac=pol,
        sync_r_lag0=r0, sync_best_lag_s=lag, sync_r_best=rbest))
    del d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = pathlib.Path(args.wesad), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ("inventory", "segments", "schedule", "quest", "cond", "qc", "qc_glitch")}
    t0 = time.time()
    subs = sorted(src.glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    for p in subs:
        audit_subject(p, p.with_name(p.stem + "_quest.csv"), rows)
        print(f"   done {p.stem} ({time.time() - t0:.0f} s)", flush=True)
    files = dict(inventory="inventory.csv", segments="label_segments.csv", schedule="schedule_check.csv",
                 quest="questionnaires.csv", cond="condition_features.csv", qc="subject_qc.csv", qc_glitch="glitches.csv")
    for k, fn in files.items():
        pd.DataFrame(rows[k]).to_csv(out / fn, index=False)
    print("tables written to", out)


if __name__ == "__main__":
    main()
