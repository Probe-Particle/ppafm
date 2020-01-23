#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

# ======= STEP 1 : Generate force-field grid 

# calculation without DFT electrostatics using atomic charges
python3 $PPPATH/generateLJFF.py -i Gr6x6N3hole.xyz 
python3 $PPPATH/generateElFF_point_charges.py -i Gr6x6N3hole.xyz --tip s


# ======= STEP 2 : Relax Probe Particle using that force-field grid 

#python3 $PPPATH/relaxed_scan.py -k 0.5 --qrange -0.05 0.0 2 --pos
python3 $PPPATH/relaxed_scan.py -k 0.5 -q -0.05
#python3 $PPPATH/relaxed_scan.py -k 0.5 -q -10.5

# ======= STEP 3 : Plot the results

#python3 $PPPATH/plot_results.py -k 0.5 --qrange -0.05 0.0 2 --arange 0.5 2.0 2 --pos --df 
python3 $PPPATH/plot_results.py -k 0.5 -q -0.05 -a 2.0 --df
#python3 $PPPATH/plot_results.py -k 0.5 -q -10.5 -a 2.0 --df


