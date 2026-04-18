## GitHub Figure Sequence

This note records a compact four-figure structure that can be rebuilt from the scripts in this repository.

### Figure 1. Reconstruction Trajectory Panel
- Goal:
  - show observed cells, predicted cells, and dense rollout trajectories in one consistent panel
- Typical use:
  - the first qualitative check after training
  - the main geometry view before criticality or downstream interpretation
- Main script path:
  - `piuot/plot.py`

### Figure 2. Potential Landscape State Map
- Goal:
  - render a readable latent-space `3D` potential terrain
  - optionally add a second panel with coarse state labels
- Typical use:
  - visualize basin geometry, transition regions, and branch topology
- Main script path:
  - downstream-specific builders can be assembled from `downstream/analyze_manifold_physics_fates.py`

### Figure 3. Multi-Model Comparison
- Goal:
  - compare PIUOT against external baselines with readable panels instead of raw metric tables
- Typical content:
  - `W1`
  - `W2^2`
  - `MMD`
  - shared manifold overlay
- Typical use:
  - benchmark figure for GitHub or paper supplements

### Figure 4. Additive Criticality
- Goal:
  - separate `action` and potential-related components before combining them into one criticality view
- Typical content:
  - action curve
  - potential-related curve
  - additive candidates
  - selected combined overlay
- Main script paths:
  - `criticality/compute_original_qreshape_mass_indicator.py`
  - `criticality/compare_potential_related_indicators.py`

## Design Rule
- Keep the repository code-only.
- Rebuild figures from your current run.
- Do not commit generated outputs into the public repository.
