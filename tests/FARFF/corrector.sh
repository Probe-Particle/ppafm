#! /bin/bash

#PPPATH="/home/prokop/git/ProbeParticleModel"
PPPATH="../../"

#ln -s ../../examples/Generator/out3/Atoms.npy .
#ln -s ../../examples/Generator/out3/Bonds.npy .

ln -s ../../examples/Generator/formic_acid/Atoms.npy .
ln -s ../../examples/Generator/formic_acid/Bonds.npy .

#python $PPPATH/pyProbeParticle/FARFF.py
# === problem with relative imports in python-3     : see :  https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py/27876800#27876800
export PYTHONPATH=$PYTHONPATH:$PPPATH
#python -m pyProbeParticle.FARFF
#python3 -m pyProbeParticle.Corrector
#python3 -m pyProbeParticle.CorrectionLoop --job train
python3 -m pyProbeParticle.CorrectionLoop --job loop
#python3 -m pyProbeParticle.GeneratorOCL_LJC