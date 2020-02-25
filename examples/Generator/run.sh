#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

#python3 $PPPATH/pyProbeParticle/GeneratorOCL_LJC.py -Y Spheres
#python3 $PPPATH/pyProbeParticle/GeneratorOCL_LJC.py -Y D-S-H
#python3 $PPPATH/pyProbeParticle/GeneratorOCL_LJC.py -Y Bonds
#python3 $PPPATH/pyProbeParticle/GeneratorOCL_LJC.py -Y AtomRfunc
#python3 $PPPATH/pyProbeParticle/GeneratorOCL_LJC.py -Y AtomsAndBonds
#python3 $PPPATH/pyProbeParticle/GeneratorOCL_Simple.py -Y AtomsAndBonds

export PYTHONPATH=$PYTHONPATH:$PPPATH
python3 -m pyProbeParticle.GeneratorOCL_Simple