#! /bin/bash

echo " ====== STEP 0 : Download Example Data-Files "

# You should either install this: https://megatools.megous.com/
# Or download it manually from Mega.nz
echo "in order to download data files we need to install megatools "
echo "alternatively you may download it manually from mega.nz webpage and place it to proper sub-folders ./tip and ./sample "
echo "If you have Debian based linux system with apt-get this should work : "
echo "enter your sudo password ( I promiss not hacking you ;-) : "
sudo apt-get install megatools

echo "Downloading Sample data-files ... "
megadl 'https://mega.nz/file/bWhTUQDQ#7mS9E-wArUzHqOevCepckHezzO8uLLC0S1PbRDsiQfs'
megadl 'https://mega.nz/file/rPgFxYAB#kQ6J90i4qQ4LlDFKq-k8PCX0rBie75_zdReNgNYbaKY'
mkdir sample
mv CHGCAR.xsf sample
mv LOCPOT.xsf sample

echo "Downloading Tip data-files ... "
megadl 'https://mega.nz/#!2CgjyCZR!b0E-vPUp6TmkV0_uTwAYHDrMXv8mJcShYOVBZVSUYs0'
mkdir tip
mv CHGCAR.xsf tip

echo "======= STEP 1 : Generate force field grid."
ppafm-conv-rho       -s sample/CHGCAR.xsf -t tip/CHGCAR.xsf -B 1.0 -E
ppafm-generate-elff  -i sample/LOCPOT.xsf --tip_dens tip/CHGCAR.xsf --Rcore 0.7 -E --doDensity
ppafm-generate-dftd3 -i sample/LOCPOT.xsf --df_name PBE

echo "======= STEP 2 : Relax Probe Particle using that force field grid."
ppafm-relaxed-scan -k 0.25 -q 1.0 --noLJ --Apauli 18.0 --bDebugFFtot # Note the --noLJ for loading separate Pauli and vdW instead of LJ force field

echo "======= STEP 3 : Plot the results."
ppafm-plot-results -k 0.25 -q 1.0 -a 2.0 --df
