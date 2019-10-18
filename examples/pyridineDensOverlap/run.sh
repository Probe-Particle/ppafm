#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

echo " ====== STEP 0 : Download Example Data-Files "

# You should either install this: https://megatools.megous.com/
# Or download it manually wrom Mega.nz
echo "in order to download data files we need to install megatools "
echo "alternatively you may download it manually from mega.nz webpage and place it to proper sub-folders ./tip and ./sample "
echo "If you have Debian based linux system with apt-get this should work : "
echo "enter your sudo password ( I promiss not hacking you ;-) : "
sudo apt-get install megatools

echo "Downloading Sample data-files ... "
megadl 'https://mega.nz/#F!GOhnRIwY!XneVlLYhCmvp74JCCH9BRA'
mkdir sample
mv CHGCAR.xsf sample
mv LOCPOT.xsf sample

echo "Downloading Tip data-files ... "
megadl 'https://mega.nz/#!2CgjyCZR!b0E-vPUp6TmkV0_uTwAYHDrMXv8mJcShYOVBZVSUYs0'
mkdir tip
mv CHGCAR.xsf tip

echo "======= STEP 1 : Generate force-field grid "

python $PPPATH/conv_rho.py     -s sample/CHGCAR.xsf -t tip/CHGCAR.xsf --Bpower 1.2 -E
python $PPPATH/generateElFF.py -i sample/LOCPOT.xsf --tip_dens tip/CHGCAR.xsf --Rcore 0.7 -E
python $PPPATH/generateLJFF.py -i sample/CHGCAR.xsf --ffModel vdW  -E

echo "======= STEP 2 : Relax Probe Particle using that force-field grid "

python $PPPATH/relaxed_scan_PVE.py -k 0.25 -q 1.0 --Apauli 1.0 --bDebugFFtot

echo "======= STEP 3 : Plot the results "

python $PPPATH/plot_results.py -k 0.25 -q 1.0 -a 2.0 --df
