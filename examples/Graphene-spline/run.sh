#! /bin/bash

# ======= STEP 1 : Generate force-field grid 

# calculation without DFT electrostatics using atomic charges
python3 ../../generateLJFF.py -i Gr6x6N3hole.xyz
python3 ../../generateElFF_point_charges.py  -i Gr6x6N3hole.xyz

# ======= STEP 2 : Relax Probe Particle using that force-field grid 

python3 ../../relaxed_scan.py  --tipspline TipRSpline.ini
mv Qo-0.10Qc0.00K0.11 Qo-0.10Qc0.00K0.11-spline
python3 ../../relaxed_scan.py 

# ======= STEP 3 : Plot the results

python3 ../../plot_results.py --Fz --atoms --bonds --WSxM

cd Qo-0.10Qc0.00K0.11
python3 ../../../plotZcurves.py -p ../curve_points.ini
cd ..

cd Qo-0.10Qc0.00K0.11-spline
python3 ../../../plotZcurves.py -p ../curve_points.ini
cd ..

#python ../../plot_results.py -k 0.5 -q -0.00 -a 2.0 --df

# COMMENT: npy test now in PTCDA_Hartree_dz2


 
