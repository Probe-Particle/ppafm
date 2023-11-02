#!/usr/bin/env python3

import time

import numpy as np
import pyopencl as cl

import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu

oclu.init_env(i_platform=0)


def handleNegativeDensity(rho):
    Q = rho.sum()
    rho[rho < 0] = 0
    rho *= Q / rho.sum()


def make_gaussian(shape, lvec):
    xyz = np.stack(np.meshgrid(*[np.linspace(-0.5, 0.5, shape[i], endpoint=False) for i in range(3)], indexing="ij"), axis=-1)
    xyz = xyz.dot(lvec[1:].T)
    array = np.exp(-0.5 * (xyz**2).sum(axis=-1))
    shift = (-np.array(shape) / 2).astype(int)
    array = np.roll(array, shift, axis=(0, 1, 2))
    return array


def test_power():
    p = 1.5
    data = FFcl.DataGrid(np.random.rand(350, 350, 350).astype(np.float32) - 0.25, np.concatenate([np.zeros((1, 3)), np.eye(3)], axis=0))
    data.cl_array  # array to device

    array2 = data.array.copy()
    t0 = time.perf_counter()
    handleNegativeDensity(array2)
    array_exp_cpu = array2**p
    cpu_time = time.perf_counter() - t0
    print(f"Exponentiation time (CPU): {cpu_time}")

    t0 = time.perf_counter()
    data_exp_gpu = data.power_positive(p=p, in_place=True)
    FFcl.oclu.queue.finish()
    gpu_time = time.perf_counter() - t0
    print(f"Exponentiation time (GPU): {gpu_time}")
    print(f"Speed-up factor: {cpu_time/gpu_time}")

    # Test that OCL routine gives the same result as numpy
    array_exp_gpu = data_exp_gpu.array
    assert np.allclose(array_exp_cpu, array_exp_gpu)

    # Test that the original array is not modified when in_place=False
    array_orig = data.array.copy()
    data.power_positive(p=p, in_place=False)
    assert np.allclose(data.array, array_orig)


def test_tip_interp():
    # Make two different grids with a gaussian peak at the corner
    shape = (200, 200, 200)
    lvec = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]])
    rho = FFcl.TipDensity(make_gaussian(shape, lvec), lvec)

    shape2 = (150, 100, 140)
    lvec2 = np.array([[0, 0, 0], [12, 0, 0], [0, 8, 0], [0, 0, 15]])
    rho2 = FFcl.TipDensity(make_gaussian(shape2, lvec2), lvec2)

    # Interpolate the first grid onto the second grid
    rho_interp = rho.interp_at(lvec2, shape2)

    # Check that the interpolant is close to the analytical solution
    assert np.allclose(rho_interp.array, rho2.array, atol=1e-3, rtol=1e-3)
