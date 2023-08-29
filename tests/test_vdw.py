#!/usr/bin/env python3

'''
Compare the C++ and OpenCL implementations of the vdW calculation and check that they are consistent.
'''

import numpy as np
import pyopencl as cl

import ppafm.common as PPU
import ppafm.core as PPC
import ppafm.HighLevel as PPH
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
        [     x_min ,    0,    0],
        [x_max-x_min,    0,    0],
        [          0, step,    0],
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

def test_dftd3():

    oclu.init_env(i_platform=0)

    Z_pp = 8
    Zs = np.array([1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 7])
    xyzs = np.array([
        [12.503441, 10.00046 ,  0.0],
        [11.21048 , 12.163759,  0.0],
        [11.21132 ,  7.83666 ,  0.0],
        [ 8.71074 , 12.06802 ,  0.0],
        [ 8.71154 ,  7.93144 ,  0.0],
        [11.41128 , 10.000239,  0.0],
        [10.69736 , 11.2     ,  0.0],
        [10.697821,  8.800241,  0.0],
        [ 9.300119, 11.145101,  0.0],
        [ 9.30056 ,  8.85458 ,  0.0],
        [ 8.59812 ,  9.999701,  0.0]]
    )
    lvec = np.array([
        [ 0,  0,  0],
        [20,  0,  0],
        [ 0, 20,  0],
        [ 0,  0, 20]
    ])
    pixPerAngstrome = 10
    params = {'s6': 1.000, 's8': 0.7875, 'a1':  0.4289, 'a2': 4.4407}

    forcefield = FFcl.ForceField_LJC()
    forcefield.initSampling(lvec, pixPerAngstrome=pixPerAngstrome)
    forcefield.prepareBuffers(atoms=np.concatenate([xyzs, np.zeros((len(Zs), 1))], axis=1), Zs=Zs)
    forcefield.setPP(Z_pp)

    forcefield.initialize()
    forcefield.add_dftd3(params)
    FF_ocl = forcefield.downloadFF()

    coeffs_ocl = np.empty((forcefield.nAtoms, 4), dtype=np.float32)
    cl.enqueue_copy(forcefield.queue, coeffs_ocl, forcefield.cl_cD3)

    coeffs_cpp = PPC.computeD3Coeffs(xyzs, Zs, Z_pp, params)
    PPU.params['gridN'] = forcefield.nDim
    PPU.params['gridA'] = lvec[1]
    PPU.params['gridB'] = lvec[2]
    PPU.params['gridC'] = lvec[3]
    FF_cpp, E_cpp = PPH.prepareArrays(None, True)
    PPC.setFF_shape(np.shape(FF_cpp), lvec)
    PPC.getDFTD3FF(xyzs, coeffs_cpp)

    FF_cpp = FF_cpp.transpose((2, 1, 0, 3))
    E_cpp = E_cpp.transpose((2, 1, 0))

    assert np.allclose(coeffs_ocl, coeffs_cpp)
    assert np.allclose(FF_ocl[..., :3], FF_cpp, rtol=1e-4, atol=1e-6)
    assert np.allclose(FF_ocl[..., 3], E_cpp, rtol=1e-4, atol=1e-6)
