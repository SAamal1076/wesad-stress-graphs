"""M1 - cross-protocol-order check of the shortcut models.

WESAD has two protocol orders: A = Base > Fun > Medi1 > TSST > Medi2 (8 subjects),
B = Base > TSST > Medi > Fun > Medi (7 subjects). Because the schedule is almost
identical within an order, 'minutes since start' identifies the condition.
Training on one order and testing on the other removes that shortcut.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.windows import build_window_index, task_view  # noqa: E402
from wesad.evaluation import summarise  # noqa: E402

OUT = ROOT / "results" / "m1"


def protocol_orders(win):
    pos = win[win.label == 2].groupby("subject").run_pos.first()
    return set(pos[pos == 4].index), set(pos[pos == 2].index)  # A: TSST 4th, B: TSST 2nd


def main():
    win = build_window_index(60.0, 5.0)
    mot = pd.read_csv(OUT / "motion_features_60s.csv", index_col=0)
    order_a, order_b = protocol_orders(win)
    rows = []
    for task in ("binary", "three_class"):
        t = task_view(win, task)
        idx = win.index[win.label.isin(t.label.unique())]
        for fname, X in {"time": t[["t_mid_min"]].values, "motion": mot.loc[idx].values}.items():
            ests = {"LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                    "RF": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced",
                                                 n_jobs=-1, random_state=0)}
            for mname, est in ests.items():
                pred = np.empty(len(t), int)
                for tr_set, te_set in ((order_a, order_b), (order_b, order_a)):
                    tr, te = t.subject.isin(tr_set).values, t.subject.isin(te_set).values
                    pred[te] = est.fit(X[tr], t.y.values[tr]).predict(X[te])
                s = summarise(t.y.values, pred, t.subject.values)
                rows.append(dict(task=task, features=fname, model=mname, **s))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cross_order_shortcuts.csv", index=False)
    print(df[["task", "features", "model", "macro_f1_mean", "bal_acc_mean", "pooled_acc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
