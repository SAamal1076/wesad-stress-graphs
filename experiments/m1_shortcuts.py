"""M1 - window index and 'shortcut' baselines under leave-one-subject-out.

Shortcut models deliberately see NO physiology. They measure how much of the task
can be solved by things that are not stress:
  * majority      : always predict the most frequent training class
  * time          : minutes since the start of the recording (baseline is always first)
  * motion        : accelerometer statistics (posture, movement, speaking/gesturing)
Any physiological model must be judged against these numbers, not against 50 %.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.windows import SUBJECTS, build_window_index, load_subject, task_view  # noqa: E402
from wesad.evaluation import per_subject_metrics, run_loso, summarise  # noqa: E402

OUT = ROOT / "results" / "m1"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0


def motion_features(windows: pd.DataFrame) -> pd.DataFrame:
    """Per-window accelerometer statistics, chest (g) and wrist (1/64 g -> g)."""
    feats = []
    for s in SUBJECTS:
        d = load_subject(s)
        dev = {"chest": (d["chest_ACC"].astype(np.float64), float(d["chest_ACC_fs"])),
               "wrist": (d["wrist_ACC"].astype(np.float64) / 64.0, float(d["wrist_ACC_fs"]))}
        w = windows[windows.subject == s]
        for r in w.itertuples():
            row = {"idx": r.Index}
            for name, (acc, fs) in dev.items():
                seg = acc[int(round(r.start_s * fs)):int(round(r.end_s * fs))]
                mag = np.linalg.norm(seg, axis=1)
                for k, ax in enumerate("xyz"):
                    row[f"{name}_mean_{ax}"] = seg[:, k].mean()   # posture (gravity direction)
                    row[f"{name}_std_{ax}"] = seg[:, k].std()     # movement
                row[f"{name}_mag_mean"] = mag.mean()
                row[f"{name}_mag_std"] = mag.std()
            feats.append(row)
    return pd.DataFrame(feats).set_index("idx").loc[windows.index]


def models():
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                n_jobs=-1, random_state=SEED)
    return {"LR": lr, "RF": rf}


def main():
    # window index for the four window lengths of RQ4 (step 5 s)
    for L in (10, 20, 30, 60):
        w = build_window_index(float(L), 5.0)
        w.to_csv(OUT / f"windows_{L}s.csv", index=False)
        print(f"windows {L:>2d}s: {len(w)} rows | per label: {w.label.value_counts().sort_index().to_dict()}")

    win = build_window_index(60.0, 5.0)
    print("computing motion features ...", flush=True)
    mot = motion_features(win)
    mot.to_csv(OUT / "motion_features_60s.csv")

    rows, per_subj = [], []
    for task in ("binary", "three_class"):
        t = task_view(win, task)
        idx = win.index[win.label.isin(t.label.unique())]
        y, g = t.y.values, t.subject.values
        feature_sets = {
            "majority": (np.zeros((len(t), 1)), {"majority": DummyClassifier(strategy="most_frequent")}),
            "time_since_start": (t[["t_mid_min"]].values, models()),
            "motion_chest+wrist": (mot.loc[idx].values, models()),
            "motion_wrist": (mot.loc[idx].filter(like="wrist").values, models()),
        }
        for fs_name, (X, mdl) in feature_sets.items():
            for m_name, est in mdl.items():
                pred = run_loso(est, X, y, g)
                s = summarise(y, pred, g)
                rows.append(dict(task=task, features=fs_name, model=m_name, **s))
                ps = per_subject_metrics(y, pred, g)
                ps.insert(0, "model", m_name); ps.insert(0, "features", fs_name); ps.insert(0, "task", task)
                per_subj.append(ps)
                print(f"{task:11s} {fs_name:20s} {m_name:8s} F1 {s['macro_f1_mean']:.3f}±{s['macro_f1_sd']:.3f} "
                      f"balAcc {s['bal_acc_mean']:.3f} | pooled acc {s['pooled_acc']:.3f} F1 {s['pooled_macro_f1']:.3f}",
                      flush=True)
    pd.DataFrame(rows).to_csv(OUT / "shortcut_baselines.csv", index=False)
    pd.concat(per_subj).to_csv(OUT / "shortcut_baselines_per_subject.csv", index=False)


if __name__ == "__main__":
    main()
