## PIUOT GitHub Release

This folder is a GitHub-ready PIUOT code bundle for arbitrary datasets.

Structure:
- `piuot/`: trajectory reconstruction
  - `train.py`: main training/reconstruction entry
  - `evaluate.py`: trajectory fit metrics
  - `plot.py`: trajectory projection plots
  - `diagnose.py`: mass diagnostics
  - `input/`: user `.h5ad` input location
  - `core/`: internal PIUOT model implementation
- `criticality/`: critical indicator design and plotting
- `downstream/`: downstream fate and perturbation plotting
- `piuot/configs/default.yaml`: shared training config

What is intentionally excluded:
- trained checkpoints
- saved figures
- output folders
- `.h5ad` data

Usage:
1. Put your `.h5ad` input under `piuot/input/`.
2. Edit `piuot/configs/default.yaml`.
3. Set `device.type` to `mps`, `cuda`, or `cpu`.
4. Choose your latent setting in `reduction.method` and `reduction.epoch`.
5. If your `.h5ad` already uses a custom embedding name, set `data.embedding_key` directly.
6. Set `data.time_key` and `data.raw_time_key`.
7. Train and reconstruct trajectories:
   - `python piuot/train.py --config piuot/configs/default.yaml`

Criticality and downstream:
- no YAML is used there
- open the corresponding `.py` script and manually edit the run name, data path, label, checkpoint, and device defaults near the top
- then run the script directly, for example:
  - `python criticality/compute_original_qreshape_mass_indicator.py`
  - `python criticality/compare_potential_related_indicators.py`
  - `python downstream/run_downstream.py`
