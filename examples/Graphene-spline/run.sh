#! /bin/bash

# ======= STEP 1 : Generate force-field grid 

# calculation without DFT electrostatics using atomic charges
python ../../generateLJFF.py -i Gr6x6N3hole.xyz
python ../../generateElFF_point_charges.py  -i Gr6x6N3hole.xyz

# ======= STEP 2 : Relax Probe Particle using that force-field grid 

python ../../relaxed_scan.py  --tipspline TipRSpline.ini
mv Qo-0.10Qc0.00K0.11 Qo-0.10Qc0.00K0.11-spline
python ../../relaxed_scan.py 

# ======= STEP 3 : Plot the results

python ../../plot_results.py --Fz --atoms --bonds --WSxM

cd Qo-0.10Qc0.00K0.11
python ../../../plotZcurves.py -p ../curve_points.ini
cd ..

cd Qo-0.10Qc0.00K0.11-spline
python ../../../plotZcurves.py -p ../curve_points.ini
cd ..

#python ../../plot_results.py -k 0.5 -q -0.00 -a 2.0 --df

# COMMENT: npy test now in PTCDA_Hartree_dz2
#echo ""
#echo "!!! Now trying the same with saving to npy !!!:"
#echo ""
#
#
#python ../../generateLJFF.py -i Gr6x6N3hole.xyz -f npy
#python ../../relaxed_scan.py -k 0.2 -q -0.00 --tipspline TipRSpline.ini -f npy
#mv Q-0.00K0.20 Q-0.00K0.20-spline
#python ../../relaxed_scan.py -k 0.2 -q -0.00 -f npy
#cd Q-0.00K0.20
#../../../plotZcurves.py -p ../curve_points.ini -f npy
#cd ..
#cd Q-0.00K0.20-spline
#../../../plotZcurves.py -p ../curve_points.ini -f npy
#cd ..


 
