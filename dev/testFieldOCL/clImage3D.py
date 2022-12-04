#!/usr/bin/python

import sys
import time
import pyopencl as cl
import numpy as np

CL_SOURCE = '''
//__constant sampler_t sampler_1 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t sampler_1 = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
__kernel void getValInPoints(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float *      vals
){
    const float4 coord     = points[get_global_id(0)];
    vals[get_global_id(0)] = read_imagef(imgIn, sampler_1, coord).x;
}

__kernel void test_kernel(
    __global  float4*      points,
    __global  float *      vals
){
    vals[get_global_id(0)]    = points[get_global_id(0)].y;
}
'''

def makeTestGrid( pmin=(-1.0,-1.0,-1.0), pmax=(1.0,1.0,1.0), n=(100,100,100) ):
    xs=np.linspace(pmin[0],pmax[0],n[0])
    ys=np.linspace(pmin[1],pmax[1],n[1])
    zs=np.linspace(pmin[2],pmax[2],n[2])
    Xs,Ys,Zs = np.meshgrid(xs,ys,zs)
    R2 = Xs**2 + Ys**2 + Zs**2
    return np.sin(R2*10).astype(np.float32).copy()

plats   = cl.get_platforms()
ctx     = cl.Context(properties=[(cl.context_properties.PLATFORM, plats[0])], devices=None)
#ctx    = cl.create_some_context()
queue   = cl.CommandQueue(ctx)
prg     = cl.Program(ctx, CL_SOURCE).build()

#src = numpy.fromstring(img.bits().asstring(img.byteCount()), dtype=numpy.uint8)
#src.shape = h, w, _ = img.height(), img.width(), 4

mf       = cl.mem_flags

#fmt      = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
#src_buf  = cl.image_from_array(ctx, src, 4)
#dest_buf = cl.Image(ctx, mf.WRITE_ONLY, fmt, shape=(w, h))
#prg.convert(queue, (w, h), None, src_buf, dest_buf, numpy.int32(w), numpy.int32(h))
#dest = numpy.empty_like(src)
#cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(w, h))
#np.zeros((256,256,256,4)).astype(np.float32)

F = makeTestGrid( )

print("F.shape", F.shape)
print(F[10,10,:])
import matplotlib.pyplot as plt
plt.imshow(F[0,:,:]); plt.colorbar(); plt.show()

ts        = np.linspace(0.0,1.0,100)
poss      = np.zeros((len(ts),4), dtype=np.float32)
poss[:,0] = 0.5 + ts *  0.2
poss[:,1] = 0.5 + ts *  3.3
poss[:,2] = 0.5 + ts * -0.2
vals      = np.zeros(len(ts), dtype=np.float32)

cl_ImgIn = cl.image_from_array(ctx,F,num_channels=1,mode='r')
cl_poss  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=poss  ) # float4
cl_vals  = cl.Buffer(ctx, mf.WRITE_ONLY                   , poss.nbytes/4 ) # float

prg.getValInPoints( queue, (len(ts),), None, *(cl_ImgIn, cl_poss, cl_vals) )

#prg.test_kernel( queue, (len(ts),), None, *(cl_poss, cl_vals) )
cl.enqueue_copy( queue, vals, cl_vals )
queue.finish()

print(vals)
