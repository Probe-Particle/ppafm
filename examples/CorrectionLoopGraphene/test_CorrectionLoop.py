
import sys
sys.path.append('../../')

from pyProbeParticle import oclUtils     as oclu 
from pyProbeParticle import fieldOCL     as FFcl 
from pyProbeParticle import RelaxOpenCL  as oclr
from pyProbeParticle import common       as PPU
from pyProbeParticle import basUtils
from pyProbeParticle.AFMulatorOCL_Simple  import AFMulator
from pyProbeParticle.GeneratorOCL_Simple2 import InverseAFMtrainer
from pyProbeParticle.AuxMap import AuxMaps
import pyProbeParticle.AuxMap as AuxMap
import pyProbeParticle.CorrectionLoop as cloop

import os
import time
import numpy as np
import matplotlib.pyplot as plt

class ExampleTrainer(InverseAFMtrainer):

    def on_sample_start(self):
        self.randomize_distance(delta=0.2)
        self.randomize_tip(max_tilt=0.3)
        # print('Distance, tipR0:', self.distAboveActive, self.afmulator.tipR0)

# =============== Setup

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    #from optparse import OptionParser
    #parser = OptionParser()
    #parser.add_option( "-j", "--job", action="store", type="string", help="[train/loop]")
    #(options, args) = parser.parse_args()

    print( " UNIT_TEST START : CorrectionLoop ... " )
    #import atomicUtils as au

    print("# ------ Init Generator   ")

    i_platform = 0
    env = oclu.OCLEnvironment( i_platform = i_platform )
    FFcl.init(env)
    oclr.init(env)

    afmulator = AFMulator(
        pixPerAngstrome = 10,
        lvec            = np.array([
                            [ 0.0,  0.0, 0.0],
                            [20.0,  0.0, 0.0],
                            [ 0.0, 20.0, 0.0],
                            [ 0.0,  0.0, 5.0]
                          ]),
        scan_window     = ((2.0, 2.0, 5.0), (18.0, 18.0, 8.0)),
    )

    #atoms = AuxMap.AtomRfunc(scan_dim=(128, 128), scan_window=((2,2),(18,18)))
    #bonds = AuxMap.Bonds(scan_dim=(128, 128), scan_window=((2,2),(18,18)))

    cloop.Job_CorrectionLoop_SimpleRandom( afmulator, geom_fname="input.xyz", geom_fname_ref="ref.xyz", nstep=1000, plt=plt )

    #print( "UNIT_TEST is not yet written :-( " )
    print( " UNIT_TEST CorrectionLoop DONE !!! " )
