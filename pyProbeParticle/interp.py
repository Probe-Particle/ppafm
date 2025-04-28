#!/usr/bin/env python3

import numpy as np
import os
import ctypes
from ctypes import c_void_p, c_int, c_double, c_bool
from .cpp_utils import compile_and_load, work_dir, _np_as, c_double_p, c_int_p
from . import utils as ut

verbosity = 0 

# Load the library
lib = compile_and_load( name='interp', bASAN=False )

# int interpolate_2d( int mode, double* data_points, double* data_vals, int ndata, double* gps, double* out_vals, int ngps, double Rcut, int nNeighMax=8, int* out_neighs=0, double* out_weights=0 ){
lib.interpolate_2d.argtypes  = [ c_int, c_double_p, c_double_p, c_int, c_double_p, c_double_p, c_int, c_double, c_int, c_int_p, c_double_p] 
lib.interpolate_2d.restype   = None
def interpolate_2d( data_points, data_vals, gps, Rcut=1.0, nNeighMax=8, bBasis=False, mode=1 ):
    ndata = len(data_points)
    ngps  = len(gps)
    print( "interpolate_2d() ngps", ngps, " ndata", ndata )
    out_vals = np.zeros(ngps)
    if bBasis:
        out_neighs  = np.zeros((ngps,nNeighMax), dtype=np.int32)
        out_weights = np.zeros((ngps,nNeighMax))
    else:
        out_neighs  = None
        out_weights = None
    lib.interpolate_2d( mode, _np_as(data_points,c_double_p), _np_as(data_vals,c_double_p), ndata, _np_as(gps,c_double_p), _np_as(out_vals,c_double_p), ngps, Rcut, nNeighMax, _np_as(out_neighs,c_int_p), _np_as(out_weights,c_double_p) )
    return out_vals, out_neighs, out_weights

# sample_kernel_func( int kind,  int n, double* xs, double* vals ){
lib.sample_kernel_func.argtypes  = [ c_int, c_int, c_double_p, c_double_p, c_double ] 
lib.sample_kernel_func.restype   = None
def sample_kernel_func(  xs, kind=1, Rcut=1.0 ):
    n = len(xs)
    vals = np.zeros(n)
    lib.sample_kernel_func( kind, n, _np_as(xs,c_double_p), _np_as(vals,c_double_p), Rcut )
    return vals

# #void interpolate_local_kriging_ok( const double* data_points, const double* data_vals, int ndata, const double* gps, double* out_vals, int ngps, double Rcut, int nNeighMax=8, int* out_neighs=0, double* out_weights=0 ){
# lib.intrepolate_local_kriging_ok.argtypes  = [c_double_p, c_double_p, c_int, c_double_p, c_double_p, c_int, c_double] 
# lib.intrepolate_local_kriging_ok.restype   = None
# def interpolate_local_kriging_ok( data_points, data_vals, gps, Rcut=1.0, nNeighMax = 8, bBasis=False ):
#     ndata = len(data_points)
#     ngps  = len(gps)
#     result_vals = np.zeros(ngps)
#     if bBasis:
#         out_neighs  = np.zeros((ngps,nNeighMax), dtype=np.int32)
#         out_weights = np.zeros((ngps*nNeighMax))
#     else:
#         out_neighs  = None
#         out_weights = None
#     lib.intrepolate_local_kriging_ok( _np_as(data_points,c_double_p), _np_as(data_vals,c_double_p), ndata, _np_as(gps,c_double_p), _np_as(out_vals,c_double_p), ngps, Rcut, nNeighMax, _np_as(out_neighs,c_int_p), _np_as(out_weights,c_double_p) )
#     return result_vals, out_neighs, out_weights
