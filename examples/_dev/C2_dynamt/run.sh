#! /bin/bash

# ======= STEP 1 : Generate force-field grid 

# calculation without DFT electrostatics using atomic charges
python ../../generateLJFF.py -i C2.xyz

# ======= STEP 2 : Relax Probe Particle using that force-field grid 

python ../../relaxed_scan.py -k 0.5 -q 0.00 --vib 3

# ======= STEP 3 : Plot the results

python ../../plot_results.py -k 0.5 -q 0.00 -a 2.0 --iets 16.0 0.0017 0.001 --cbar

 
