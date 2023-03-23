
: ======= STEP 1 : Generate force-field grid

: calculation without DFT electrostatics using atomic charges
ppafm-generate-ljff -i Gr6x6N3hole.xyz
ppafm-generate-elff-point-charges -i Gr6x6N3hole.xyz --tip s

: ======= STEP 2 : Relax Probe Particle using that force-field grid

ppafm-relaxed-scan -k 0.5 -q -0.05

: ======= STEP 3 : Plot the results

ppafm-plot-results -k 0.5 -q -0.05 -a 2.0 --df
