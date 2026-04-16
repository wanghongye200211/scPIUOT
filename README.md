## PIUOT GitHub Release

This folder is a GitHub-ready PIUOT code bundle for arbitrary datasets.

Structure:
- `piuot/`: trajectory reconstruction and evaluation
- `criticality/`: critical indicator design and plotting
- `downstream/`: downstream fate and perturbation plotting
- `configs/default.yaml`: shared experiment config

What is intentionally excluded:
- trained checkpoints
- saved figures
- output folders
- `.h5ad` data

Usage:
1. Put your `.h5ad` input under `piuot/data/input/`.
2. Edit `configs/default.yaml`.
3. Set `device.train`, `device.analysis`, and `device.perturbation` to `mps`, `cuda`, or `cpu`.
4. Choose your latent setting in `reduction.method` and `reduction.epoch`.
5. If your `.h5ad` already uses a custom embedding name, set `data.embedding_key` directly.
6. Set `data.time_key`, `data.raw_time_key`, and optional `data.state_key / data.fate_key`.
7. Run:
   - `python piuot/run_piuot.py --yaml-config configs/default.yaml`
   - `python criticality/compute_original_qreshape_mass_indicator.py --yaml-config configs/default.yaml`
   - `python criticality/compare_potential_related_indicators.py --yaml-config configs/default.yaml`
   - `python downstream/run_downstream.py --yaml-config configs/default.yaml`
