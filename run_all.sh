#!/usr/bin/env bash
# Full pipeline, in milestone order. Expects the unzipped dataset in data/WESAD/S2 ... S17.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

echo "M0  prepare and audit"
python scripts/prepare_wesad.py --wesad data/WESAD --out data/wesad_small
python scripts/eda_audit.py --wesad data/WESAD --out results/eda
python scripts/eda_figures.py

echo "M1  windows, leakage tests, shortcut baselines, cross-order check"
python -m pytest -q tests
python experiments/m1_shortcuts.py
python experiments/m1_cross_order.py

echo "M2  physiological features, coupling graphs, descriptive effects"
python experiments/m2_features.py --win 60
python experiments/m2_descriptive.py

echo "M3  classifiers (LOSO and cross-order) and hypothesis tests"
for task in binary three_class; do
  for scheme in cross_order loso; do
    python experiments/m3_models.py --task "$task" --scheme "$scheme"
  done
done
python experiments/m3_stats.py

echo "M4  personalisation, no-clock sensitivity, G2 test-time graph smoothing"
for task in binary three_class; do
  for scheme in cross_order loso; do
    python experiments/m4_personalise.py --task "$task" --scheme "$scheme"
  done
done
python experiments/m4_g2.py
python experiments/m4_stats.py

echo "M5  interpretation"
python experiments/m5_interpret.py

echo "M6  learned total-variation graphs (B: per-window joint model; C: online graph and decoupling index)"
python experiments/m6_gate.py                 # synthetic gate for eta and theta (amendment 1)
python experiments/m6_tv_features.py          # per-window TV graph on every WESAD window
python experiments/m6_descriptive.py
python experiments/m6c_gate.py                # synthetic gate for mu
python experiments/m6c_online.py --mu 0.25    # causal online graph over each whole recording
python experiments/m6c_evaluate.py            # indices, H10-H11, latency, H8/H9/H12, figures
echo "M7  report figures, visual panels and the one-page summary (needs playwright + Chrome)"
python experiments/m7_report_figures.py
python experiments/m7_visuals.py
python experiments/m7_portfolio.py
echo "done"
