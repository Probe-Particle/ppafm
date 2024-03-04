#!/bin/bash

# KPFM on a CH3Br (methylbromide) molecule with a Xe-terminated tip.
# Simulates AFM for three different values of bias voltage (negative, zero, and positive) and calculates LCPD based on that.
# Besides the standard KPFM simulation, calculates LCPD arising from two separate terms in KPFM:
# (1) static charge in the sample (CH3Br) interacting with the charge induced in the tip due to its polarizability by the bias (tVs0), and
# (2) charge induced in the sample (CH3Br) by the bias (due to the molecules polarizability) interacting with static charge on the Xe tip (t0sV)

#The input Hartree potentials are files in the (Gaussian) "cube" format: hartree_potential_V0.cube (without an external field) and hartree_potential_Vz.cube (in the homogeneous field of Vref=+0.1 V/angstrom). These files have been generated with FHI-AIMS [https://fhi-aims.org/] and can be downloaded from the Zenodo site, record #10562922:
echo "Download Hartree Potentials"
for f in hartree_potential_V{0,z}.cube ; do
    if [ ! -f $f ] ; then
	wget --no-check-certificate "https://zenodo.org/records/10562922/files/$f"
    fi
done

echo "Calculate Electrostatic Forces and Polarizability in External Field"
ppafm-generate-elff -i hartree_potential_V0.cube -t s --KPFM_sample hartree_potential_Vz.cube --KPFM_tip fit --Vref +0.1 --Rcore -1.0 --z0 0.0
echo

echo "Calculate Lennard-Jones Interaction"
ppafm-generate-ljff -i hartree_potential_V0.cube
echo

echo "Tip Relaxation"
ppafm-relaxed-scan --Vrange -0.5 0.5 3 --pol_t 1.0 --pol_s 1.0
echo

echo "Plotting"
ppafm-plot-results --Vrange -0.5 0.5 3 --LCPD_maps --cbar --V0 0 --save_df --df
echo

echo
echo "LCPD decomposition: Prepare directories"
for d in tVs0 t0sV ; do
   if [ -d $d ] ; then
    \rm -fr ${d}/Q*K* ${d}/*.xsf
   else
       if [ -f $d ] ; then
	   \rm -f $d
       fi
       mkdir $d
   fi
   \cp -f params.ini ${d}/
   \ln -f FF*.xsf ${d}/
done

echo
echo "LCPD decomposition: Induced charge on the tip with static charge of the sample"
cd tVs0

echo "Tip Relaxation"
ppafm-relaxed-scan --Vrange -0.5 0.5 3 --pol_t 1.0 --pol_s 0.0
echo

echo "Plotting"
ppafm-plot-results --Vrange -0.5 0.5 3 --LCPD_maps --cbar --V0 0
echo
cd ..

echo
echo "LCPD decomposition: Static charge on the tip with induced charge of the sample"
cd t0sV

echo "Tip Relaxation"
ppafm-relaxed-scan --Vrange -0.5 0.5 3 --pol_t 0.0 --pol_s 1.0
echo

echo "Plotting"
ppafm-plot-results --Vrange -0.5 0.5 3 --LCPD_maps --cbar --V0 0
echo
cd ..

echo "All done"
exit
