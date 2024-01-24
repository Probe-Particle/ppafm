#!/bin/bash

echo "Download Hartree Potentials"
wget --no-check-certificate "https://zenodo.org/records/10563098/files/KPFM_hartree.tar.gz"

echo "Extract Hartree Potentials"
tar xzvf KPFM_hartree.tar.gz

echo "Calculate Electrostatic Forces and Polarizability in External Field"
ppafm-generate-elff -i LOCPOT_V0.xsf -t dz2 --KPFM_sample LOCPOT_Vz.xsf --KPFM_tip fit --Vref +0.1 --Rcore -1.0 --z0 0.0
echo

echo "Calculate Lennard-Jones Interaction"
ppafm-generate-ljff -i LOCPOT_V0.xsf
echo

echo "Tip Relaxation"
ppafm-relaxed-scan --Vrange -0.5 0.5 3
echo

echo "Plotting"
ppafm-plot-results --Vrange -0.5 0.5 3 --LCPD_maps --cbar --V0 0 --df --atoms
echo

echo "All done"
exit
