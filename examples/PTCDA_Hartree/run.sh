#! /bin/bash
wget --no-check-certificate "https://www.dropbox.com/s/18eg89l89npll8x/LOCPOT.xsf.zip"
unzip LOCPOT.xsf.zip

# ======= STEP 1 : Generate force-field grid
ppafm-generate-elff -i LOCPOT.xsf
ppafm-generate-ljff -i LOCPOT.xsf

# ======= STEP 2 : Relax Probe Particle using that force-field grid
ppafm-relaxed-scan -k 0.5 --qrange -0.10 0.10 3 --pos

# ======= STEP 3 : Plot the results
ppafm-plot-results -k 0.5 --qrange -0.10 0.10 3 --arange 0.5 2.0 2 --pos --df
