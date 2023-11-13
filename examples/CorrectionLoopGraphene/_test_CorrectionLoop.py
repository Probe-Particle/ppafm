import time

import matplotlib.pyplot as plt
import numpy as np

import ppafm.atomicUtils as au
import ppafm.ml.AuxMap as AuxMap
import ppafm.SimplePot as pot
from ppafm import common as PPU
from ppafm import io
from ppafm.ml.AuxMap import AuxMaps
from ppafm.ml.CorrectionLoop import CorrectionLoop
from ppafm.ml.Corrector import Corrector, Molecule
from ppafm.ml.Generator import InverseAFMtrainer
from ppafm.ocl import field as FFcl
from ppafm.ocl import oclUtils as oclu
from ppafm.ocl import relax as oclr
from ppafm.ocl.AFMulator import AFMulator


def Job_CorrectionLoop_SimpleRandom(simulator, geom_fname="input.xyz", geom_fname_ref="ref.xyz", nstep=10, plt=None):
    """
    Correction loop which does not use any Force-Field nor AuxMap or Neural-Network
    it simply randomly add/remove or move atoms
    """
    corrector = Corrector()
    corrector.logImgName = "AFM_Err"
    corrector.xyzLogFile = "CorrectorLog.xyz"
    corrector.plt = plt
    corrector.izPlot = -1
    nscan = simulator.scan_dim
    nscan = (nscan[0], nscan[1], nscan[2] - len(simulator.dfWeight))
    simulator.scan_window

    def makeMol(fname):
        xyzs, Zs, qs, _ = io.loadXYZ(fname)
        xyzs[:, 0] += -2
        xyzs[:, 1] += -8 + 20.0
        xyzs[:, 2] += -2.2
        # AFMulatorOCL.setBBoxCenter( xyzs, [0.0,0.0,0.0] )
        # scan_center = np.array([sw[1][0] + sw[0][0], sw[1][1] + sw[0][1]]) / 2
        # print( "scan_center ", scan_center )
        # xyzs[:,:2] += scan_center - xyzs[:,:2].mean(axis=0)
        # xyzs[:,2]  += (sw[1][2] - 9.0) - xyzs[:,2].max()
        # print("xyzs ", xyzs)
        xyzqs = np.concatenate([xyzs, qs[:, None]], axis=1)
        molecule = Molecule(xyzs, Zs, qs)
        return molecule

    mol_ref = makeMol(geom_fname_ref)
    # simulator.bSaveFF = True                #    DEBUG !!!!!!!!!!!!!!!!!
    simulator.saveFFpre = "ref_"
    AFMs = simulator(mol_ref.xyzs, mol_ref.Zs, mol_ref.qs)
    simulator.saveFFpre = ""
    np.save("AFMref.npy", AFMs)

    AFMRef = np.load("AFMref.npy")
    # AFMRef = np.roll( AFMRef,  5, axis=0 );
    # AFMRef = np.roll( AFMRef, -6, axis=1 );

    looper = CorrectionLoop(None, simulator, None, None, corrector)
    looper.plt = plt
    # looper.logImgName = "CorrectionLoopAFMLog"
    # looper.logAFMdataName = "AFMs"

    molecule = makeMol(geom_fname)

    looper.startLoop(molecule, None, None, None, AFMRef)
    ErrConv = 0.01
    print("# ------ To Loop    ")
    # exit()
    for itr in range(nstep):
        # print( "# ======= CorrectionLoop[ %i ] ", itr )
        Err = looper.iteration(itr=itr)
        if Err < ErrConv:
            break


# =============== Setup

if __name__ == "__main__":
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    # from optparse import OptionParser
    # parser = OptionParser()
    # parser.add_option( "-j", "--job", action="store", type="string", help="[train/loop]")
    # (options, args) = parser.parse_args()

    print(" UNIT_TEST START : CorrectionLoop ... ")
    # import atomicUtils as au

    print("# ------ Init Generator   ")

    i_platform = 0
    env = oclu.OCLEnvironment(i_platform=i_platform)
    FFcl.init(env)
    oclr.init(env)

    afmulator = AFMulator(
        pixPerAngstrome=10,
        lvec=np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 6.0]]),
        scan_window=((2.0, 2.0, 5.0), (18.0, 18.0, 8.0)),
    )

    # atoms = AuxMap.AtomRfunc(scan_dim=(128, 128), scan_window=((2,2),(18,18)))
    # bonds = AuxMap.Bonds(scan_dim=(128, 128), scan_window=((2,2),(18,18)))

    pot.init_random(int(45446 * time.time()))
    Job_CorrectionLoop_SimpleRandom(afmulator, geom_fname="input.xyz", geom_fname_ref="ref.xyz", nstep=1000, plt=plt)

    # print( "UNIT_TEST is not yet written :-( " )
    print(" UNIT_TEST CorrectionLoop DONE !!! ")
