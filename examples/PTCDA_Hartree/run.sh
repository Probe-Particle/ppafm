#! /bin/bash

if [ ! -f "LOCPOT.xsf" ] ; then
    wget "https://zenodo.org/records/14222456/files/PTCDA_Ag.zip"
    unzip PTCDA_Ag.zip
    rm PTCDA_Ag.zip
fi

# ======= STEP 1 : Generate force-field grid
ppafm-generate-elff -i LOCPOT.xsf
ppafm-generate-ljff -i LOCPOT.xsf

# ======= STEP 2 : Relax Probe Particle using that force-field grid
ppafm-relaxed-scan -k 0.5 --qrange -0.10 0.10 3 --pos

# ======= STEP 3 : Plot the results
ppafm-plot-results -k 0.5 --qrange -0.10 0.10 3 --arange 0.5 2.0 2 --pos --df
