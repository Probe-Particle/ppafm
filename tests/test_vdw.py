#!/usr/bin/env python3

'''
Compare the C++ and OpenCL implementations of the vdW calculation and check that they are consistent.
'''


import numpy as np

import ppafm.common as PPU
import ppafm.core as PPC
import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu


def test_vdw():

    oclu.init_env(i_platform=0)

    Z_pp = 8
    Z_atom = [1, 6, 7, 8]
    x_min, x_max, step = 0.3, 6.0, 0.1
    x_FF = np.arange(x_min, x_max, step)
    atoms = np.zeros((1, 4))
    lvec = np.array([
        [     x_min ,    0,   0],
        [x_max-x_min,    0,   0],
        [          0, step,   0],
        [          0,    0, step]
    ])
    pixPerAngstrome = round(1 / step)

    forcefield = FFcl.ForceField_LJC()
    typeParams = PPU.loadSpecies('atomtypes.ini')
    REAs = PPU.getAtomsREA(Z_pp, Z_atom, typeParams)
    cLJs = PPU.REA2LJ(REAs)

    forcefield.initSampling(lvec, pixPerAngstrome=pixPerAngstrome)

    for damp_method in [-1, 0, 1, 2, 3, 4]:

        for i, Z in enumerate(Z_atom):

            cLJs_ = cLJs[i:i+1]
            REAs_ = REAs[i:i+1]
            forcefield.prepareBuffers(atoms, cLJs_, REAs=REAs_)
            forcefield.initialize()

            forcefield.addvdW(damp_method=damp_method)
            FF_ocl = forcefield.downloadFF()
            Fx_ocl = FF_ocl[:, 0, 0, 0]
            E_ocl = FF_ocl[:, 0, 0, 3]

            coefs = cLJs_[0] if damp_method in [-1, 0] else REAs_[0]
            E_cpp, Fx_cpp = PPC.evalRadialFF(-x_FF, coefs, kind=damp_method)

            assert np.allclose(Fx_ocl, Fx_cpp, atol=1e-6, rtol=1e-4)
            assert np.allclose(E_ocl, E_cpp, atol=1e-6, rtol=1e-4)

test_vdw()
