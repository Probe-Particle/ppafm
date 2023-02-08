#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

# ======= STEP 1 : Generate force-field grid

python3 $PPPATH/generateLJFF.py -i PTCDA.xyz
python3 $PPPATH/generateElFF_point_charges.py -i PTCDA.xyz --tip s

# ======= STEP 2 : Relax Probe Particle using that force-field grid

#python3 $PPPATH/relaxed_scan.py -k 0.5 --qrange -0.10 0.10 3 --pos
python3 $PPPATH/relaxed_scan.py -k 0.5 -q -0.10

# ======= STEP 3 : Plot the results

#python3 $PPPATH/plot_results.py -k 0.5 --qrange -0.10 0.10 3 --arange 0.5 2.0 2 --pos --df
python3 $PPPATH/plot_results.py -k 0.5 -q -0.10 --arange 0.5 2.0 2 --df
