#! /bin/bash

# ======= STEP 1 : Generate force-field grid
# calculation without DFT electrostatics using atomic charges
ppafm-generate-ljff -i dichlor-brom-benzene.xyz -f npy
ppafm-generate-elff-point-charges -i dichlor-brom-benzene.xyz --tip dz2 -f npy

# ======= STEP 2 : Relax Probe Particle using that force-field grid
ppafm-relaxed-scan -k 0.5 -q -0.05 -f npy

# ======= STEP 3 : Plot the results
ppafm-plot-results -k 0.5 -q -0.05 -a 0.5 --df -f npy
