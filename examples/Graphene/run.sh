#! /bin/bash

# ======= STEP 1 : Generate force-field grid 

# calculation without DFT electrostatics using atomic charges
python ../../generateLJFF.py -i Gr6x6N3hole.xyz -q

# ALTERNATIVELY : calculation with DFT electrostatics
#python ../../generateLJFF.py -i Gr6x6N3hole.xyz
#python ../../generateElFF.py LOCPOT.xsf

# ======= STEP 2 : Relax Probe Particle using that force-field grid 

#python ../../relaxed_scan.py -k 0.5 --qrange -0.05 0.0 2 --pos
python ../../relaxed_scan.py -k 0.5 -q -0.05

# ======= STEP 3 : Plot the results

#python ../../plot_results.py -k 0.5 --qrange -0.05 0.0 2 --arange 0.5 2.0 2 --pos --df 
python ../../plot_results.py -k 0.5 -q -0.05 -a 2.0 --df



 