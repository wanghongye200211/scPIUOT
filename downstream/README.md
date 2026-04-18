## Downstream Notes

This folder contains manual downstream plotting and summary scripts.

Files:
- `run_downstream.py`: convenience wrapper to execute the full downstream chain
- `analyze_manifold_physics_fates.py`: physics-fate panel and related summaries
- `build_focus_bundle.py`: compact focus bundle for a selected run
- `build_perturbation_dynamic_fraction.py`: perturbation-driven dynamic cell-type fractions
- `build_perturbation_manifest.py`: collect perturbation outputs into a lightweight manifest
- `GITHUB_FIGURE_SEQUENCE.md`: compact four-figure layout for GitHub or paper assembly

How to use:
1. Open `run_downstream.py` or the target script.
2. Edit the manual defaults near the top:
   - run name
   - data path
   - embedding key
   - checkpoint
   - output label / slug
   - state / fate keys
   - device
3. Run the script directly.

Latest validated downstream workflow:
- use this folder as a manual downstream workflow template
- choose your own run, embedding, labels, and checkpoint
- this folder is intended for rebuilding manuscript figures from your current run rather than shipping fixed result bundles

Common figure sequence:
- `Figure 1`: reconstruction trajectory panel
- `Figure 2`: potential landscape state map
- `Figure 3`: multi-model benchmark board
- `Figure 4`: additive criticality panel

Design rule:
- keep this folder as a code-only manual workflow
- do not rely on YAML here
- do not commit generated figures or manifests into this GitHub release
