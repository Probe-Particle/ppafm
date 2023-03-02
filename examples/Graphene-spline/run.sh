#! /bin/bash
PPAFM_DIR='../../ppafm/cmdline'

# ======= STEP 1 : Generate force-field grid

ppafm-generate-ljff -i Gr6x6N3hole.xyz

# ======= STEP 2 : Relax Probe Particle using that force-field grid

ppafm-relaxed-scan -k 0.5 -q -0.00 --tipspline TipRSpline.ini
mv Q-0.00K0.50 Q-0.00K0.50-spline
ppafm-relaxed-scan -k 0.5 -q -0.00

# ======= STEP 3 : Plot the results

cd Q-0.00K0.50
python ../${PPAFM_DIR}/utilities/plotZcurves.py -p ../curve_points.ini
cd ..

cd Q-0.00K0.50-spline
python ../${PPAFM_DIR}/utilities/plotZcurves.py -p ../curve_points.ini
cd ..

#ppafm-plot-results -k 0.5 -q -0.00 -a 2.0 --df

# --- npy not working for now at all !!! issue#53 should be added here!
#echo ""
#echo "!!! Now trying the same with saving to npy !!!:"
#echo ""
#
#
#ppafm-generate-ljff -i Gr6x6N3hole.xyz -f npy
#ppafm-relaxed-scan -k 0.2 -q -0.00 --tipspline TipRSpline.ini -f npy
#mv Q-0.00K0.20 Q-0.00K0.20-spline
#ppafm-relaxed-scan -k 0.2 -q -0.00 -f npy
#cd Q-0.00K0.20
#${PPAFM_DIR}/../plotZcurves.py -p ../curve_points.ini -f npy
#cd ..
#cd Q-0.00K0.20-spline
#${PPAFM_DIR}/../plotZcurves.py -p ../curve_points.ini -f npy
#cd ..
