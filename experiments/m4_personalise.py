"""M4a - personalisation: centre each subject's features on its own first C minutes of baseline.

The calibration period is the only thing the test subject contributes (unlabelled except for the
knowledge that the person was asked to rest, as in a VR calibration phase). Windows starting inside
the calibration period are excluded from evaluation for every model.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from wesad.windows import task_view  # noqa: E402
from m3_models import feature_sets, inner_select, make_lr, outer_splits  # noqa: E402
from wesad.evaluation import assert_no_subject_leak  # noqa: E402

OUT = ROOT / "results" / "m4"
(OUT / "oof").mkdir(parents=True, exist_ok=True)
CLOCK = ["temp_chest_mean", "temp_wrist_mean", "eda_chest_scl_mean", "eda_wrist_scl_mean"]
CONFIGS = [("both", "levels"), ("both", "levels+rw"), ("both", "levels+pcorr"),
           ("chest", "levels"), ("chest", "levels+rw"), ("wrist", "levels"), ("wrist", "levels+rw"),
           ("both", "levels-noclock"), ("both", "levels+rw-noclock")]


def calibration(df, minutes):
    """Boolean masks: calibration windows, and windows to exclude from evaluation."""
    b0 = df[df.label == 1].groupby("subject").start_s.min()      # start of the baseline run
    end_cal = df.subject.map(b0) + 60.0 * minutes
    calib = (df.label == 1) & (df.end_s <= end_cal + 1e-9)
    exclude = df.start_s < end_cal - 1e-9
    return calib.values, exclude.values


def personalise(df, cols, calib):
    out = df.copy()
    mu = df.loc[calib].groupby("subject")[cols].mean()
    out[cols] = df[cols].values - mu.loc[df.subject].values
    return out


def fold(df, y, g, tr, te, cands):
    assert_no_subject_leak(g, tr, te)
    k, C = inner_select(df.iloc[tr], y[tr], g[tr], cands)
    m = make_lr(C).fit(df.iloc[tr][cands[k]].values, y[tr])
    return te, m.predict_proba(df.iloc[te][cands[k]].values), k, C


def main(task, scheme, jobs):
    feats = pd.read_csv(ROOT / "results" / "m2" / "features_60s.csv")
    for minutes in (2, 5):
        calib_all, excl_all = calibration(feats, minutes)
        feats[f"calib{minutes}"], feats[f"excl{minutes}"] = calib_all, excl_all
    meta = []
    for sensor, name in CONFIGS:
        base = name.replace("-noclock", "")
        cands = feature_sets(feats.columns, sensor)[base]
        if name.endswith("-noclock"):
            cands = [[c for c in cl if c not in CLOCK] for cl in cands]
        all_cols = sorted(set(sum(cands, [])))
        modes = ["none", "calib2"] + (["calib5"] if (sensor, name) == ("both", "levels+rw") else [])
        for mode in modes:
            t0 = time.time()
            df = personalise(feats, all_cols, feats.calib2.values if mode == "calib2" else feats.calib5.values) \
                if mode != "none" else feats
            t = task_view(df, task)
            y, g = t.y.values, t.subject.values
            res = Parallel(n_jobs=jobs)(delayed(fold)(t, y, g, tr, te, cands) for tr, te in outer_splits(g, scheme))
            prob = np.zeros((len(y), len(np.unique(y))))
            for te, p, _, _ in res:
                prob[te] = p
            tag = f"{task}__{scheme}__{sensor}__{name}__{mode}"
            np.savez_compressed(OUT / "oof" / f"{tag}.npz", y=y, subject=g, start_s=t.start_s.values,
                                end_s=t.end_s.values, run_id=t.run_id.values, label=t.label.values,
                                excl2=t.excl2.values, excl5=t.excl5.values, prob=prob)
            ev = ~t[f"excl{5 if mode == 'calib5' else 2}"].values
            f1 = f1_score(y[ev], prob[ev].argmax(1), average="macro")
            meta.append(dict(tag=tag, chosen=[(int(k), C) for _, _, k, C in res]))
            print(f"{tag:62s} pooled F1 (eval windows) {f1:.3f} ({time.time() - t0:.0f} s)", flush=True)
    json.dump(meta, open(OUT / f"meta_{task}_{scheme}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    main(a.task, a.scheme, a.jobs)
