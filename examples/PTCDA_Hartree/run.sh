#! /bin/bash

PPAFM_DIR='../../ppafm/cmdline'

wget --no-check-certificate "https://www.dropbox.com/s/18eg89l89npll8x/LOCPOT.xsf.zip"
unzip LOCPOT.xsf.zip

# ======= STEP 1 : Generate force-field grid
python ${PPAFM_DIR}/generateElFF.py -i LOCPOT.xsf
python ${PPAFM_DIR}/generateLJFF.py -i LOCPOT.xsf

# ======= STEP 2 : Relax Probe Particle using that force-field grid
python ${PPAFM_DIR}/relaxed_scan.py -k 0.5 --qrange -0.10 0.10 3 --pos

# ======= STEP 3 : Plot the results

python ${PPAFM_DIR}/plot_results.py -k 0.5 --qrange -0.10 0.10 3 --arange 0.5 2.0 2 --pos --df
