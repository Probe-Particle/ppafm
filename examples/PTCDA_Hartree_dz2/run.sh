#! /bin/bash

PPAFM_DIR='../../ppafm/cmdline'

wget --no-check-certificate "https://www.dropbox.com/s/18eg89l89npll8x/LOCPOT.xsf.zip"
unzip LOCPOT.xsf.zip

# ======= STEP 1 : Generate force-field grid
python ${PPAFM_DIR}/generateElFF.py -i LOCPOT.xsf --tip dz2
ppafm-generate-ljff -i LOCPOT.xsf

# ======= STEP 2 : Relax Probe Particle using that force-field grid
ppafm-relaxed-scan -k 0.5 -q -0.10

# ======= STEP 3 : Plot the results
ppafm-plot-results -k 0.5 -q -0.10 -a 2.0 2 --df
