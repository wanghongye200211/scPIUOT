## Criticality Notes

These scripts are manual analysis templates.

Files:
- `compute_original_qreshape_mass_indicator.py`: build raw `Q_reshape^mass`, action, drift, and product curves
- `compare_potential_related_indicators.py`: compare multiple potential-related indicators from the same run

How to use:
1. Open the target script.
2. Edit the manual defaults near the top:
   - run name
   - seed
   - checkpoint selector
   - device
   - output label / slug
3. Run the script directly.

Suggested criticality views:
- original `Q_reshape^mass`
- action curve
- drift-related companion curve when needed
- product-style criticality
- additive criticality with separate component curves

What is not bundled here:
- figures
- saved manifests
- previous downstream outputs

This repository only keeps the code path. Recreate results by pointing the script to your own run directory.
