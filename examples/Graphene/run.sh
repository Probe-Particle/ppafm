#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

# ======= STEP 1 : Generate force-field grid 

# calculation without DFT electrostatics using atomic charges
python $PPPATH/generateLJFF.py -i Gr6x6N3hole.xyz 
python $PPPATH/generateElFF_point_charges.py -i Gr6x6N3hole.xyz 

# ======= STEP 2 : Relax Probe Particle using that force-field grid 

python $PPPATH/relaxed_scan.py -k 0.5 --qrange -0.05 0.0 2 --pos
#python $PPPATH/relaxed_scan.py -k 0.5 -q -0.05
#python $PPPATH/relaxed_scan.py -k 0.5 -q -0.5

# ======= STEP 3 : Plot the results

python $PPPATH/plot_results.py -k 0.5 --qrange -0.05 0.0 2 --arange 0.5 2.0 2 --pos --df 
#python $PPPATH/plot_results.py -k 0.5 -q -0.05 -a 2.0 --df
#python $PPPATH/plot_results.py -k 0.5 -q -0.5 -a 2.0 --df

 
