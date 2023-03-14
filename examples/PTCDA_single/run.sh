#! /bin/bash

# ======= STEP 1 : Generate force-field grid
ppafm-generate-ljff -i PTCDA.xyz
ppafm-generate-elff-point-charges -i PTCDA.xyz --tip s

# ======= STEP 2 : Relax Probe Particle using that force-field grid
ppafm-relaxed-scan -k 0.5 -q -0.10

# ======= STEP 3 : Plot the results
ppafm-plot-results -k 0.5 -q -0.10 --arange 0.5 2.0 2 --df
