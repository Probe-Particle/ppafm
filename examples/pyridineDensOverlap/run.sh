echo "======= STEP 1 : Download the sample and tip files."
wget https://zenodo.org/records/14222456/files/pyridine.zip?download=1 -O sample.zip
wget https://zenodo.org/records/14222456/files/CO_tip.zip?download=1  -O tip.zip

echo "======= STEP 2 : Unzip the sample and tip files."
unzip sample.zip
unzip tip.zip -d tip

echo "======= STEP 3 : Generate force field grid."
ppafm-conv-rho       -s CHGCAR.xsf -t tip/density_CO.xsf -B 1.0 -E
ppafm-generate-elff  -i LOCPOT.xsf --tip_dens tip/density_CO.xsf --Rcore 0.7 -E --doDensity
ppafm-generate-dftd3 -i LOCPOT.xsf --df_name PBE

echo "======= STEP 4 : Relax Probe Particle using that force field grid."
ppafm-relaxed-scan -k 0.25 -q 1.0 --noLJ --Apauli 18.0 --bDebugFFtot # Note the --noLJ for loading separate Pauli and vdW instead of LJ force field

echo "======= STEP 5 : Plot the results."
ppafm-plot-results -k 0.25 -q 1.0 -a 2.0 --df
