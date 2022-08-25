#!/usr/bin/env python3

import sys
import time
import numpy as np
import pyopencl as cl

sys.path.append('..')
import pyProbeParticle.fieldOCL  as FFcl
import pyProbeParticle.oclUtils  as oclu

def handleNegativeDensity( rho ):
    Q = rho.sum()
    rho[rho<0] = 0
    rho *= ( Q/rho.sum() )

oclu.init_env()

p = 1.5
shape = (350, 350, 350)
array = np.random.rand(*shape).astype(np.float32)

array_orig = array.copy()

t0 = time.perf_counter()
handleNegativeDensity(array)
array_exp = array ** p
cpu_time = time.perf_counter() - t0
print(f'Exponentiation time (CPU): {cpu_time}')

ctx = FFcl.oclu.ctx
queue = FFcl.oclu.queue
mf = cl.mem_flags
array_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=array_orig)
queue.finish()

t0 = time.perf_counter()
array_cl = FFcl.runPower(array_cl, p=p, queue=queue)
queue.finish()
gpu_time = time.perf_counter() - t0
print(f'Exponentiation time (GPU): {gpu_time}')
print(f'Speed-up factor: {cpu_time/gpu_time}')

array_exp2 = np.empty_like(array)
cl.enqueue_copy(queue, array_exp2, array_cl)
queue.finish()

assert np.allclose(array_exp, array_exp2)
