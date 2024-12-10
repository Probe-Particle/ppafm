#! /bin/bash

# An example of using the full-density based model (FDBM) with an sample electron density obtained with FHI-aims.
# The elecron density in aims is all-electron, so the density at the nuclei positions can be extremely high.
# This sometimes causes artifacts to appear in the resuting simulation images. This can be mitigated by cutting
# out the high electron density values, which is demonstrated in this example.

# Download input data
tip_dir="./tip"
sample_dir="./sample"
if [ ! -d "$tip_dir" ]; then
    wget https://zenodo.org/records/10563098/files/CO_tip_densities.tar.gz
    mkdir $tip_dir
    tar -xvf CO_tip_densities.tar.gz --directory $tip_dir
fi
if [ ! -d "$sample_dir" ]; then
    wget "https://zenodo.org/records/14222456/files/pyridineBrCl.zip?download=1" -O hartree.zip
    mkdir $sample_dir
    unzip hartree.zip -d $sample_dir
fi

# Generate the force field
# Notice here the `--density_cutoff` option. Try removing it and observe how artifacts appear in the image at the top-right
# at the position of the bromine atom.
ppafm-conv-rho       -s sample/density.xsf -t tip/density_CO.xsf -B 1.1 --density_cutoff 100
ppafm-generate-elff  -i sample/hartree.xsf --tip dz2
ppafm-generate-dftd3 -i sample/hartree.xsf --df_name PBE

# Relax the probe particle in the force field
ppafm-relaxed-scan -k 0.24 -q -0.1 --noLJ --Apauli 25.0

# Plot the results
ppafm-plot-results -k 0.24 -q -0.1 -a 1.0 --df
