#!/usr/bin/env python3

import time

import numpy as np

import ppafm.common as PPU
import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu

R2SAFE = 1e-6
ADamp_Const = 180.0
ADamp_R2 = 0.5
ADamp_R4 = 0.5
ADamp_invR4 = 0.03
ADamp_invR8 = 0.01

def vdw_nodamp(p, xyzs, Zs, cLJs):
    vdw = np.zeros(p.shape[:3] + (4,), dtype=np.float32)
    for xyz, Z, c in zip(xyzs, Zs, cLJs):
        dp = p - xyz
        ir2 = 1 / ((dp ** 2).sum(axis=-1) + R2SAFE)
        ir6 = ir2 ** 3
        E = -c[0] * ir6
        F = (6 * E * ir2)[..., None] * dp
        vdw[..., :3] += F
        vdw[..., 3] += E
    return vdw

def vdw_damp_constant(p, xyzs, Zs, cLJs):
    vdw = np.zeros(p.shape[:3] + (4,), dtype=np.float32)
    for xyz, Z, c in zip(xyzs, Zs, cLJs):
        dp = p - xyz
        ir2 = 1 / ((dp ** 2).sum(axis=-1) + ADamp_Const * c[0])
        ir6 = ir2 ** 3
        E = -c[0] * ir6
        F = (6 * E * ir2)[..., None] * dp
        vdw[..., :3] += F
        vdw[..., 3] += E
    return vdw

def get_vdw(damp_method, xyzs, Zs, lvec, nDim, cLJs):

    p_xyz = [np.linspace(lvec[0][i], lvec[i+1][i], nDim[i], endpoint=False) for i in range(3)] # Assuming an orthogonal grid here
    p = np.stack(np.meshgrid(*p_xyz, indexing='ij'), axis=-1).astype(np.float32)

    if damp_method == -1:
        vdw = vdw_nodamp(p, xyzs, Zs, cLJs)
    elif damp_method == 0:
        vdw = vdw_damp_constant(p, xyzs, Zs, cLJs)
    else:

        vdw = np.zeros(p.shape[:3] + (4,), dtype=np.float32)
        for xyz, Z, c in zip(xyzs, Zs, cLJs):

            R0 = np.float32((2*c[1]/c[0]) ** (1/6))
            E0 = np.float32(-c[0]**2 / (4 * c[1]))
            dp = p - xyz
            r2 = (dp ** 2).sum(axis=-1)
            iR2 = np.float32(1 / (R0 ** 2))
            u2 = r2 * iR2
            u4 = u2 * u2

            if damp_method == 1:
                D = np.zeros_like(r2, dtype=np.float32); dD = np.zeros_like(r2, dtype=np.float32)
                D[u2 <= 1] = 1 - u2[u2 <= 1]
                dD[u2 <= 1] = -2
            elif damp_method == 2:
                D = np.zeros_like(r2, dtype=np.float32); dD = np.zeros_like(r2, dtype=np.float32)
                D[u2 <= 1] = (1 - u2[u2 <= 1]) ** 2
                dD[u2 <= 1] = -4 * (1 - u2[u2 <= 1])
            elif damp_method == 3:
                D = (1 / (u2 + R2SAFE)) ** 2
                dD = -4 * (1 / (u2 + R2SAFE)) ** 3
            elif damp_method == 4:
                D = (1 / (u2 + R2SAFE)) ** 4
                dD = -8 * (1 / (u2 + R2SAFE)) ** 5
            else:
                raise ValueError(f'Invalid vdW damp method {damp_method}')

            e = 1 / (u4 * u2 + D*ADamp_R2)
            E = 2 * E0 * e
            F = (E * e * (6 * u4 + dD * ADamp_R2) * iR2)[..., None] * dp
            vdw[..., :3] += F
            vdw[..., 3] += E

    return vdw

def test_vdw():

    oclu.init_env(i_platform=0)

    n_atom = 10
    xyzs = 20 * np.random.rand(n_atom, 3).astype(np.float32)
    Zs = np.random.randint(1, 16, n_atom)
    qs = np.zeros(n_atom, dtype=np.float32)
    atoms = np.concatenate([xyzs, qs[:, None]], axis=1)
    lvec = np.array([
        [ 0,  0,  0],
        [20,  0,  0],
        [ 0, 20,  0],
        [ 0,  0, 20]
    ])
    pixPerAngstrome = 5

    forcefield = FFcl.ForceField_LJC()
    typeParams = PPU.loadSpecies('atomtypes.ini')
    REAs = PPU.getAtomsREA(8, Zs, typeParams, alphaFac=-1.0)
    cLJs = PPU.REA2LJ(REAs)

    forcefield.initSampling(lvec, pixPerAngstrome=pixPerAngstrome)
    forcefield.prepareBuffers(atoms, cLJs, REAs=REAs)

    for damp_method in [0, 1, 2, 3, 4]:

        print('Damp method:', damp_method)

        forcefield.initialize(bFinish=True)

        t0 = time.perf_counter()
        forcefield.addvdW(damp_method=damp_method)
        forcefield.queue.finish()
        print(f'Calc time: {time.perf_counter() - t0}')

        FF_ocl = forcefield.downloadFF()
        FF_np = get_vdw(damp_method, xyzs, Zs, lvec, forcefield.nDim, cLJs)

        assert np.allclose(FF_ocl, FF_np, rtol=1e-4, atol=1e-6)

if __name__ == '__main__':
    test_vdw()
