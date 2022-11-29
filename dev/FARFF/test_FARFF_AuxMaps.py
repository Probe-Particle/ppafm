import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
sys.path.append('../../')

#from pyProbeParticle import oclUtils     as oclu 
#from pyProbeParticle import fieldOCL     as FFcl 
#from pyProbeParticle import RelaxOpenCL  as oclr
#from pyProbeParticle import common       as PPU
#from pyProbeParticle import basUtils
#from pyProbeParticle.AFMulatorOCL_Simple  import AFMulator
#from pyProbeParticle.GeneratorOCL_Simple2 import InverseAFMtrainer
#from pyProbeParticle.AuxMap import AuxMaps
#import pyProbeParticle.AuxMap as AuxMap
import pyProbeParticle.atomicUtils as au
import pyProbeParticle.FARFF as fff


# =============== Setup

if __name__ == "__main__":

    from pyProbeParticle import basUtils
    import pyProbeParticle.atomicUtils as au
    import pyProbeParticle.GLView as glv

    xyzs, Zs, qs, _ = basUtils.loadXYZ("input.xyz")

    atomMapF, bondMapF, lvecMap = fff.makeGridFF( fff )
    print( " atomMapF ", atomMapF.shape, " bondMapF ", bondMapF.shape,  )

    relaxer = fff.EngineFARFF()

    relaxer.preform_relaxation( molecule=None, xyzs=xyzs, Zs=Zs, qs=qs, lvec=lvecMap, atomMap=atomMapF, bondMap=bondMapF, Fconv=-1e-5 )


