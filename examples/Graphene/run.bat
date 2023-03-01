
set PPPATH="../.."
set PPAFM_RECOMPILE=1

: ======= STEP 1 : Generate force-field grid

: calculation without DFT electrostatics using atomic charges
python %PPPATH%/generateLJFF.py -i Gr6x6N3hole.xyz
python %PPPATH%/generateElFF_point_charges.py -i Gr6x6N3hole.xyz --tip s

: ======= STEP 2 : Relax Probe Particle using that force-field grid

python %PPPATH%/relaxed_scan.py -k 0.5 -q -0.05

: ======= STEP 3 : Plot the results

python %PPPATH%/plot_results.py -k 0.5 -q -0.05 -a 2.0 --df
