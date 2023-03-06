import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../')

#from ppafm import oclUtils     as oclu
#from ppafm import fieldOCL     as FFcl
#from ppafm import RelaxOpenCL  as oclr
#from ppafm import common       as PPU
#from ppafm.AFMulatorOCL_Simple  import AFMulator
#from ppafm.GeneratorOCL_Simple2 import InverseAFMtrainer
#from ppafm.AuxMap import AuxMaps
#import ppafm.AuxMap as AuxMap
import ppafm.atomicUtils as au
import ppafm.FARFF as fff

# =============== Setup

if __name__ == "__main__":

    import ppafm.atomicUtils as au
    import ppafm.GLView as glv
    from ppafm import io

    xyzs, Zs, qs, _ = io.loadXYZ("input.xyz")

    atomMapF, bondMapF, lvecMap = fff.makeGridFF( fff )
    print( " atomMapF ", atomMapF.shape, " bondMapF ", bondMapF.shape,  )

    relaxer = fff.EngineFARFF()

    relaxer.preform_relaxation( molecule=None, xyzs=xyzs, Zs=Zs, qs=qs, lvec=lvecMap, atomMap=atomMapF, bondMap=bondMapF, Fconv=-1e-5 )
