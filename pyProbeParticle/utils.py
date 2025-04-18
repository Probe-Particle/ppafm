#!/usr/bin/env python3

import numpy as np

def makePosXY(n=100, L=10.0, axs=(0,1,2), p0=(0.0,0.0,0.0) ):
    x = np.linspace(-L,L,n)
    y = np.linspace(-L,L,n)
    Xs,Ys = np.meshgrid(x,y)
    ps = np.zeros((n*n,3))
    ps[:,axs[0]] = p0[axs[0]] + Xs.flatten()
    ps[:,axs[1]] = p0[axs[1]] + Ys.flatten()
    ps[:,axs[2]] = p0[axs[2]] 
    return ps, Xs, Ys

def makeCircle( n=10, R=1.0, p0=(0.0,0.0,0.0), axs=(0,1,2), phi0=0.0 ):
    phis  = np.linspace(0,2*np.pi,n, endpoint=False) + phi0
    ps    = np.zeros((n,3))
    ps[:,axs[0]] = p0[axs[0]] + np.cos(phis)*R
    ps[:,axs[1]] = p0[axs[1]] + np.sin(phis)*R
    ps[:,axs[2]] = p0[axs[2]]
    return ps, phis
    
def makeRotMats(phis, nsite=3 ):
    rot = np.zeros((nsite,3,3))
    ca = np.cos(phis)
    sa = np.sin(phis)
    rot[:,0,0] = ca
    rot[:,1,1] = ca
    rot[:,0,1] = sa
    rot[:,1,0] = -sa
    rot[:,2,2] = 1.0
    return rot