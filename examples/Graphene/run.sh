#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

# ======= STEP 1 : Generate force-field grid 

# calculation without DFT electrostatics using atomic charges
python $PPPATH/generateLJFF.py -i Gr6x6N3hole.xyz 
python $PPPATH/generateElFF_point_charges.py -i Gr6x6N3hole.xyz 

# ALTERNATIVELY : calculation with DFT electrostatics
#python $PPPATH/generateElFF.py -i LOCPOT.xsf
#python $PPPATH/generateLJFF.py -i LOCPOT.xsf

# ======= STEP 2 : Relax Probe Particle using that force-field grid 

#python $PPPATH/relaxed_scan.py -k 0.5 --qrange -0.05 0.0 2 --pos
python $PPPATH/relaxed_scan.py -k 0.5 -q -0.05

# ======= STEP 3 : Plot the results

#python $PPPATH/plot_results.py -k 0.5 --qrange -0.05 0.0 2 --arange 0.5 2.0 2 --pos --df 
python $PPPATH/plot_results.py -k 0.5 -q -0.05 -a 2.0 --df

echo ""
echo "!!! Now trying the same with saving to npy !!!:"
echo ""

python $PPPATH/generateLJFF.py -i Gr6x6N3hole.xyz  -f npy
python $PPPATH/generateElFF_point_charges.py -i Gr6x6N3hole.xyz  -f npy
python $PPPATH/relaxed_scan.py -k 0.2 -q -0.05 -f npy 
python $PPPATH/plot_results.py -k 0.2 -q -0.05 -a 2.0 --df -f npy


 
