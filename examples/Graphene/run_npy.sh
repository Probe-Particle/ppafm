#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

# ======= STEP 1 : Generate force-field grid 

# calculation without DFT electrostatics using atomic charges
python $PPPATH/generateLJFF.py -i Gr6x6N3hole.xyz  -f npy 
python $PPPATH/generateElFF_point_charges.py -i Gr6x6N3hole.xyz -f npy --tip='dz2' # trying if multipoles are working

# ======= STEP 2 : Relax Probe Particle using that force-field grid 

#python $PPPATH/relaxed_scan.py -k 0.5 --qrange -0.05 0.0 2 --pos --npy
python $PPPATH/relaxed_scan.py -k 0.5 -q +0.05 -f npy
#python $PPPATH/relaxed_scan.py -k 0.5 -q -0.5 --npy

# ======= STEP 3 : Plot the results

#python $PPPATH/plot_results.py -k 0.5 --qrange -0.05 0.0 2 --arange 0.5 2.0 2 --pos --df  --npy 
python $PPPATH/plot_results.py -k 0.5 -q +0.05 -a 2.0 --df -f npy
#python $PPPATH/plot_results.py -k 0.5 -q -0.5 -a 2.0 --df  --npy


