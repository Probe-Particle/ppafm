#! /bin/bash

if [ ! -f "LOCPOT.xsf" ] ; then
    wget "https://zenodo.org/records/14222456/files/PTCDA_Ag.zip"
    unzip PTCDA_Ag.zip
    rm PTCDA_Ag.zip
fi

# ======= STEP 1 : Generate force-field grid
ppafm-generate-elff -i LOCPOT.xsf --tip dz2 -f npy
ppafm-generate-ljff -i LOCPOT.xsf -f npy

# ======= STEP 2 : Relax Probe Particle using that force-field grid
ppafm-relaxed-scan -k 0.5 -q -0.10 -f npy

# ======= STEP 3 : Plot the results
ppafm-plot-results -k 0.5 -q -0.10 -a 2.0 --df -f npy
