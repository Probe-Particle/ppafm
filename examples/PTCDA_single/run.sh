#! /bin/bash

# ======= STEP 1 : Generate force-field grid
PPAFM_RECOMPILE=1 ppafm-generate-ljff -i PTCDA.xyz
PPAFM_RECOMPILE=1 ppafm-generate-elff-point-charges -i PTCDA.xyz --tip s

# ======= STEP 2 : Relax Probe Particle using that force-field grid
PPAFM_RECOMPILE=1 ppafm-relaxed-scan -k 0.5 -q -0.10

# ======= STEP 3 : Plot the results
PPAFM_RECOMPILE=1 ppafm-plot-results -k 0.5 -q -0.10 --arange 0.5 2.0 2 --df
