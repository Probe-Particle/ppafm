#!/usr/bin/python

import sys
import os
import numpy as np
import time

import pyProbeParticle.PolyCycles  as pcff
import pyProbeParticle.atomicUtils as au

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import common    as PPU
#from optparse import OptionParser

def plotCycles(cpos=None,vpos=None,nvs=None):
    if cpos is not None:
        #print "cpos !!!! ", cpos,  cpos[:,0],cpos[:,1]
        #plt.plot(cpos[:,0],cpos[:,1],"+k")
        plt.plot(cpos[:,0],cpos[:,1],"o")
        #plt.plot([0.0,2.5],[0.0,0.7],"ob")
    if vpos is not None:
        plt.plot(vpos[:,0],vpos[:,1],".")
        if nvs is not None:
            iv=0
            for ni in nvs:
                vs=np.vstack( (vpos[iv:iv+ni,:],vpos[iv,:]) )
                plt.plot(vs[:,0],vs[:,1],".-")
                iv+=ni
    plt.axis("equal")


def findBonds(ps,Rs,fc=1.5):
    n=len(ps)
    bonds  = []
    neighs = [ [] for i in range(n) ]
    for i in range(n):
        for j in range(i+1,n):
            d=ps[j]-ps[i]
            R=((Rs[i]+Rs[j])*1.5)
            r2=np.dot(d,d) 
            if(r2<(R*R)):
                #print i,j,R,np.sqrt(r2)
                bonds.append((i,j))
                neighs[i].append(j)
                neighs[j].append(i)
    #print bonds
    return bonds, neighs

#def orderTriple(a,b,c):
#    if a>b:
#        

def tryInsertTri_i(tris,tri,i):
    if tri in tris:
        tris[tri].append(i)
    else:
        tris[tri] = [i]

def tryInsertTri(tris,tri):
    #print "tri ", tri
    if not (tri in tris):
        tris[tri] = []

def findTris(bonds,neighs):
    tris = {};
    for b in bonds:
        a_ngs = neighs[b[0]]
        b_ngs = neighs[b[1]]
        common = []
        for i in a_ngs:
            if i in b_ngs:
                common.append(i)
        #print "bond ",b," common ",common
        if len(common)>2:
            print "WARRNING: bond ", b, " share these neighbors ", common
        else:
            tri0 = tuple(sorted(b+(common[0],)))
            if len(common)==2:
                tri1 = tuple(sorted(b+(common[1],)))
                #print tri0, tri1
                tryInsertTri_i(tris,tri0,common[1])
                tryInsertTri_i(tris,tri1,common[0])
            else:
                tryInsertTri(tris,tri0)
    return tris


def trisToPoints(tris,ps):
    ops=np.empty((len(tris),2))
    for i,tri in enumerate(tris.keys()):
        ops[i,:] = (ps[tri[0],:] + ps[tri[1],:] + ps[tri[2],:])/3.0
        #ops.append()
    return ops


if __name__ == "__main__":
    N    = 20
    #nvs  = np.random.randint( 3,8, N, dtype=np.int32 );       print "nvs:   ", nvs
    nvs  = np.random.randint( 5,8, N, dtype=np.int32 );       print "nvs:   ", nvs
    rots = np.random.rand    ( N ) * np.pi*2;                 print "rots:  ", rots
    nv = pcff.setup(nvs)
    #print "nv: ", nv
    #cpos = pcff.getCpos(N)
    #vpos = pcff.getVpos(nv)
    cpos,vpos=pcff.getPos(N,nv)
    #cpos[:,:] = np.random.rand( N,2)
    #cpos[:,0]*=20.0;
    #cpos[:,1]*=20.0;
    cpos[:,0]=np.arange(N)*2.5
    cpos[:,1]=np.sin(cpos[:,0]*0.3)
    print "cpos: ", cpos
    #print "vpos: ", vpos

    import matplotlib.pyplot as plt
    #plt.figure(); plotCycles(cpos=cpos)
    
    pcff.setupOpt(dt=0.3, damping=0.05, f_limit=1.0,v_limit=1.0 )
    pcff.relaxNsteps(kind=0, nsteps=1000)

    #pcff.init(rots)
    
    Rs = 0.5/np.sin(np.pi/nvs)
    plt.scatter(cpos[:,0], cpos[:,1], s=(Rs*60)**2, c='none' )
    print "nvs ",nvs
    print "Rs ",Rs
    
    bonds, neighs  = findBonds(cpos,Rs,fc=1.0)
    print "neighs ", neighs

    tris = findTris(bonds,neighs)
    print "tris ",tris
    
    ops = trisToPoints(tris,cpos)
    print "ops", ops
    
    plt.plot( ops[:,0],ops[:,1], "*" )
    
    #exit()
    for b in bonds:
        pb=cpos[b,:]
        plt.plot( pb[:,0],pb[:,1],'-b' )
    
    plt.show()
    #pcff.relaxNsteps(nsteps=10, F2conf=-1.0, dt=0.1, damp=0.9)








