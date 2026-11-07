"""
prepare_wesad.py - shrink the WESAD dataset into small per-subject files.

What it does
------------
* Reads each subject's synchronised file (SX.pkl) one at a time,
  directly from WESAD.zip: nothing needs to be unzipped.
* Keeps every signal, converts to float32, and downsamples the 700 Hz chest
  signals (with an anti-aliasing filter) to rates that are enough for
  stress analysis. Wrist signals are already low-rate and kept as they are.
* Stores, for every signal, a label array at the same rate as the signal.
* Copies the small questionnaire (SX_quest.csv) and readme files.
* Writes one compressed file per subject: out_dir/SX.npz

Only one subject is in memory at a time (about 1-2 GB of RAM is enough).

Usage
-----
    python prepare_wesad.py --wesad path/to/WESAD.zip --out wesad_small
    (--wesad can also be the unzipped WESAD folder)

Labels (see the WESAD readme): 1 = baseline, 2 = stress, 3 = amusement,
4 = meditation; 0 and 5-7 are transient / not to be used.

Reading a subject back
----------------------
    d = np.load("wesad_small/S2.npz")
    ecg, fs, lab = d["chest_ECG"], float(d["chest_ECG_fs"]), d["chest_ECG_label"]
"""
import argparse
import io
import pathlib
import pickle
import time
import zipfile

import numpy as np
from scipy.signal import resample_poly

FS_CHEST = 700  # all chest (RespiBAN) signals and the labels are at 700 Hz

# Downsampling factor for each chest signal (700 / factor = new rate).
# None = drop the signal. Change these if you need other rates.
CHEST_DOWN = {
    "ECG": 2,     # 350 Hz: enough for R-peak detection and HRV
    "EDA": 20,    # 35 Hz
    "Resp": 20,   # 35 Hz
    "ACC": 20,    # 35 Hz
    "EMG": 2,     # 350 Hz (set to None to drop it and save space)
    "Temp": 100,  # 7 Hz
}

# Native sampling rates of the wrist (Empatica E4) signals
WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def labels_at_rate(chest_labels, fs, n):
    """Label of each sample of a signal sampled at fs (n samples)."""
    idx = (np.arange(n) * (FS_CHEST / fs)).astype(np.int64)
    idx = np.minimum(idx, len(chest_labels) - 1)
    return chest_labels[idx].astype(np.int8)


def repair_adc_glitches(chest, pad=3):
    """Linearly interpolate invalid RespiBAN samples BEFORE any filtering.

    Found on the first real run: S3 has a 0.12 s episode (83 samples, inside the
    stress condition) where every chest channel sits on its ADC rail at once
    (Temp = -273.15 C, EDA = 0, ACC = -6.6 g, Resp = -50, EMG/ECG = -1.5).
    resample_poly would smear this spike into ringing (e.g. Temp of -122 C and
    +72 C in the 7 Hz signal), so it must be removed first.
    Returns the number of repaired samples.
    """
    temp = np.asarray(chest["Temp"]).ravel()
    bad = temp < -100.0  # open-circuit thermistor -> -273.15 C
    if not bad.any():
        return 0
    idx = np.flatnonzero(bad)
    for off in range(-pad, pad + 1):  # also cover the edges of the episode
        bad[np.clip(idx + off, 0, len(bad) - 1)] = True
    bad_idx, good_idx = np.flatnonzero(bad), np.flatnonzero(~bad)
    for name in list(chest.keys()):
        x = np.array(chest[name], dtype=np.float64).reshape(len(bad), -1)
        for c in range(x.shape[1]):
            x[bad_idx, c] = np.interp(bad_idx, good_idx, x[good_idx, c])
        chest[name] = x
    return int(bad.sum())


def process_subject(fileobj, out_dir):
    d = pickle.load(fileobj, encoding="latin1")  # files were pickled with Python 2
    subject = str(d["subject"])
    labels = np.asarray(d["label"]).ravel()
    out = {}

    n_fixed = repair_adc_glitches(d["signal"]["chest"])
    if n_fixed:
        print(f"  [repair] {subject}: {n_fixed} invalid chest samples "
              f"({n_fixed / FS_CHEST:.2f} s) interpolated before resampling")

    for name, x in d["signal"]["chest"].items():
        factor = CHEST_DOWN.get(name, 20)
        if factor is None:
            continue
        x = np.asarray(x, dtype=np.float64)
        y = resample_poly(x, 1, factor, axis=0, padtype="line") if factor > 1 else x  # "line": no zero-padding edge artefact
        fs = FS_CHEST / factor
        out[f"chest_{name}"] = y.astype(np.float32)
        out[f"chest_{name}_fs"] = np.float32(fs)
        out[f"chest_{name}_label"] = labels_at_rate(labels, fs, len(y))

    for name, x in d["signal"]["wrist"].items():
        fs = WRIST_FS.get(name)
        if fs is None:
            print(f"  [warning] unknown wrist signal {name!r}, skipped")
            continue
        y = np.asarray(x, dtype=np.float32)
        out[f"wrist_{name}"] = y
        out[f"wrist_{name}_fs"] = np.float32(fs)
        out[f"wrist_{name}_label"] = labels_at_rate(labels, fs, len(y))

    out_path = out_dir / f"{subject}.npz"
    np.savez_compressed(out_path, **out)

    minutes = {LABEL_NAMES[c]: round(float(np.sum(labels == c)) / FS_CHEST / 60, 1)
               for c in LABEL_NAMES}
    size_mb = out_path.stat().st_size / 1e6
    print(f"  {subject}: {size_mb:.1f} MB written | minutes per condition: {minutes}")
    del d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wesad", required=True, help="WESAD.zip or the unzipped WESAD folder")
    ap.add_argument("--out", default="wesad_small", help="output folder")
    args = ap.parse_args()

    src = pathlib.Path(args.wesad)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            names = z.namelist()
            pkls = sorted(n for n in names if n.lower().endswith(".pkl"))
            small = [n for n in names if n.endswith("_quest.csv") or n.endswith("_readme.txt")]
            print(f"Found {len(pkls)} subject files in {src.name}")
            for n in small:  # questionnaires and readmes are tiny: copy them
                (out_dir / pathlib.Path(n).name).write_bytes(z.read(n))
            for n in pkls:
                print(f"Processing {n} ...")
                with z.open(n) as f:
                    process_subject(io.BufferedReader(f, buffer_size=1 << 20), out_dir)
    else:
        pkls = sorted(src.glob("S*/S*.pkl"))
        print(f"Found {len(pkls)} subject files in {src}")
        for p in src.glob("S*/S*_quest.csv"):
            (out_dir / p.name).write_bytes(p.read_bytes())
        for p in src.glob("S*/S*_readme.txt"):
            (out_dir / p.name).write_bytes(p.read_bytes())
        for p in pkls:
            print(f"Processing {p} ...")
            with open(p, "rb") as f:
                process_subject(f, out_dir)

    total = sum(f.stat().st_size for f in out_dir.glob("*.npz")) / 1e6
    print(f"Done in {time.time() - t0:.0f} s. Total size of {out_dir}: {total:.0f} MB")


if __name__ == "__main__":
    main()
