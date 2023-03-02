#! /bin/bash
PPAFM_DIR='../../ppafm/cmdline'

# ======= STEP 1 : Generate force-field grid

python ${PPAFM_DIR}/generateLJFF.py -i Gr6x6N3hole.xyz

# ======= STEP 2 : Relax Probe Particle using that force-field grid

python ${PPAFM_DIR}/relaxed_scan.py -k 0.5 -q -0.00 --tipspline TipRSpline.ini
mv Q-0.00K0.50 Q-0.00K0.50-spline
python ${PPAFM_DIR}/relaxed_scan.py -k 0.5 -q -0.00

# ======= STEP 3 : Plot the results

cd Q-0.00K0.50
python ../${PPAFM_DIR}/utilities/plotZcurves.py -p ../curve_points.ini
cd ..

cd Q-0.00K0.50-spline
python ../${PPAFM_DIR}/utilities/plotZcurves.py -p ../curve_points.ini
cd ..

#python ${PPAFM_DIR}/plot_results.py -k 0.5 -q -0.00 -a 2.0 --df

# --- npy not working for now at all !!! issue#53 should be added here!
#echo ""
#echo "!!! Now trying the same with saving to npy !!!:"
#echo ""
#
#
#python ${PPAFM_DIR}/generateLJFF.py -i Gr6x6N3hole.xyz -f npy
#python ${PPAFM_DIR}/relaxed_scan.py -k 0.2 -q -0.00 --tipspline TipRSpline.ini -f npy
#mv Q-0.00K0.20 Q-0.00K0.20-spline
#python ${PPAFM_DIR}/relaxed_scan.py -k 0.2 -q -0.00 -f npy
#cd Q-0.00K0.20
#${PPAFM_DIR}/../plotZcurves.py -p ../curve_points.ini -f npy
#cd ..
#cd Q-0.00K0.20-spline
#${PPAFM_DIR}/../plotZcurves.py -p ../curve_points.ini -f npy
#cd ..
