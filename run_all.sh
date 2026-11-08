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
python experiments/m7_report_figures.py
echo "done"
