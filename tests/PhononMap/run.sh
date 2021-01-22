#! /bin/bash

PPPATH="/home/prokop/git/ProbeParticleModel"
#PPPATH="../../.."

#ln -s ../../examples/Generator/out3/Atoms.npy .
#ln -s ../../examples/Generator/out3/Bonds.npy .

#ln -s ../../examples/Generator/formic_acid/Atoms.npy .
#ln -s ../../examples/Generator/formic_acid/Bonds.npy .

#python $PPPATH/photonMap.py --homo PTCDA_opt-neutral_B3LYP_HOMO_pts100.cube  --lumo PTCDA_opt-neutral_B3LYP_LUMO_pts100.cube -R 10.0 -z 5.0 -t s

python $PPPATH/photonMap.py --homo PTCDA_opt-neutral_B3LYP_HOMO_pts100.cube  --lumo PTCDA_opt-neutral_B3LYP_LUMO_pts100.cube -R 10.0 -z 5.0 -t s --excitons --volumetric

#python $PPPATH/photonMap.py --dens PTCDA_exc-B3LYP-neutral_geo-B3LYP.Tr1.cube -R 10.0 -z 5.0 -t s
