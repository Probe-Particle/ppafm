#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

#ln -s ../../examples/Generator/out3/Atoms.npy .
#ln -s ../../examples/Generator/out3/Bonds.npy .

ln -s ../../examples/Generator/formic_acid/Atoms.npy .
ln -s ../../examples/Generator/formic_acid/Bonds.npy .

python $PPPATH/pyProbeParticle/FARFF.py