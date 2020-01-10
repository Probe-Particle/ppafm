#! /bin/bash

# ======= STEP 1 : Generate force-field grid 

# calculation without DFT electrostatics using atomic charges
python3 ../../generateLJFF.py -i Gr6x6N3hole.xyz -q

# ALTERNATIVELY : calculation with DFT electrostatics
#python3 ../../generateElFF.py -i LOCPOT.xsf
#python3 ../../generateLJFF.py -i LOCPOT.xsf

# ======= STEP 2 : Relax Probe Particle using that force-field grid 

#python3 ../../relaxed_scan.py -k 0.5 --qrange -0.05 0.0 2 --pos
python3 ../../relaxed_scan.py -k 0.5 -q -0.05

# ======= STEP 3 : Plot the results

#python3 ../../plot_results.py -k 0.5 --qrange -0.05 0.0 2 --arange 0.5 2.0 2 --pos --df 
python3 ../../plot_results.py -k 0.5 -q -0.05 -a 2.0 --df

echo ""
echo "!!! Now trying the same with saving to npy !!!:"
echo ""

python3 ../../generateLJFF.py -i Gr6x6N3hole.xyz -q --npy
python3 ../../relaxed_scan.py -k 0.2 -q -0.05 --npy
python3 ../../plot_results.py -k 0.2 -q -0.05 -a 2.0 --df --npy


 
