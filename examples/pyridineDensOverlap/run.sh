#! /bin/bash

PPAFM_DIR='../../ppafm/cli'

echo " ====== STEP 0 : Download Example Data-Files "

# You should either install this: https://megatools.megous.com/
# Or download it manually wrom Mega.nz
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

echo "======= STEP 1 : Generate force-field grid "
python ${PPAFM_DIR}/conv_rho.py      -s sample/CHGCAR.xsf -t tip/CHGCAR.xsf -B 1.0 -E
python ${PPAFM_DIR}/generateElFF.py  -i sample/LOCPOT.xsf --tip_dens tip/CHGCAR.xsf --Rcore 0.7 -E --doDensity
python ${PPAFM_DIR}/generateDFTD3.py -i sample/LOCPOT.xsf --df_name PBE

echo "======= STEP 2 : Relax Probe Particle using that force-field grid "
python ${PPAFM_DIR}/relaxed_scan_PVE.py -k 0.25 -q 1.0 --Apauli 18.0 --bDebugFFtot

echo "======= STEP 3 : Plot the results "
ppafm-plot-results -k 0.25 -q 1.0 -a 2.0 --df
