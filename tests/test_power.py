#!/usr/bin/env python3

import time

import numpy as np
import pyopencl as cl

import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu


def handleNegativeDensity( rho ):
    Q = rho.sum()
    rho[rho<0] = 0
    rho *= ( Q/rho.sum() )

def test_power():

    oclu.init_env(i_platform=0)

    p = 1.5
    data = FFcl.DataGrid(
        np.random.rand(350, 350, 350).astype(np.float32) - 0.25,
        np.concatenate([np.zeros((1, 3)), np.eye(3)], axis=0)
    )
    data.cl_array # array to device

    array2 = data.array.copy()
    t0 = time.perf_counter()
    handleNegativeDensity(array2)
    array_exp_cpu = array2 ** p
    cpu_time = time.perf_counter() - t0
    print(f'Exponentiation time (CPU): {cpu_time}')

    t0 = time.perf_counter()
    data_exp_gpu = data.power_positive(p=p, in_place=True)
    FFcl.oclu.queue.finish()
    gpu_time = time.perf_counter() - t0
    print(f'Exponentiation time (GPU): {gpu_time}')
    print(f'Speed-up factor: {cpu_time/gpu_time}')

    # Test that OCL routine gives the same result as numpy
    array_exp_gpu = data_exp_gpu.array
    assert np.allclose(array_exp_cpu, array_exp_gpu)

    # Test that the original array is not modified when in_place=False
    array_orig = data.array.copy()
    data.power_positive(p=p, in_place=False)
    assert np.allclose(data.array, array_orig)
