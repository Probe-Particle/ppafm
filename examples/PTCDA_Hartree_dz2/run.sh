#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

wget --no-check-certificate "https://www.dropbox.com/s/18eg89l89npll8x/LOCPOT.xsf.zip"
unzip LOCPOT.xsf.zip

# ======= STEP 1 : Generate force-field grid 

python3 $PPPATH/generateElFF.py -i LOCPOT.xsf -f npy #--tip dz2
python3 $PPPATH/generateLJFF.py -i LOCPOT.xsf -f npy 

# ======= STEP 2 : Relax Probe Particle using that force-field grid 

python3 $PPPATH/relaxed_scan.py -f npy  #-k 0.5 --qrange -0.20 0.20 3 --pos
#python3 $PPPATH/relaxed_scan.py -k 0.5 -q -0.10 --pos

# ======= STEP 3 : Plot the results

python3 $PPPATH/plot_results.py --df -f npy #-k 0.5 --qrange -0.20 0.20 3 --arange 0.5 2.0 2 --pos --df
#python3 $PPPATH/plot_results.py -k 0.5 -q -0.10 --arange 0.5 2.0 2 --pos --df


 
