#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../.."

#python3 $PPPATH/photonMap.py -R 10.0 -Z 5.0 -t s --excitons
#python3 $PPPATH/photonMap.py -R 10.0 -Z 5.0 -t s --excitons --volumetric
#python3 $PPPATH/photonMap.py -R 10.0 -Z 5.0 -t s --excitons --volumetric --beta 1.0
python3 $PPPATH/photonMap.py -R 10.0 -Z 5.0 -t s --excitons --volumetric --beta 1.0 -m molecules4.ini -c cubefiles_dens.ini

#python3 $PPPATH/photonMap.py --homo PTCDA_opt-neutral_B3LYP_HOMO_pts100.cube  --lumo PTCDA_opt-neutral_B3LYP_LUMO_pts100.cube -R 10.0 -z 5.0 -t s --excitons --volumetric
#python3 $PPPATH/photonMap.py --dens PTCDA_exc-B3LYP-neutral_geo-B3LYP.Tr1.cube -R 10.0 -z 5.0 -t s
