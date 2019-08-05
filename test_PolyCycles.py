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


def make_anim(getData,fig,nframes,interval=200):
    from matplotlib.animation import FuncAnimation
    #scat = None
    def init():
        #global scat
        cpos,sz,vpos = getData(0)
        #print sz; exit()
        #scat  = plt.scatter(cpos[:,0], cpos[:,1], s=(sz*40)**2, c="white" )
        scat  = plt.scatter(cpos[:,0], cpos[:,1], s=(sz*34)**2, c='none' )
        vplot, = plt.plot(vpos[:,0], vpos[:,1], ".r" )
        #xs,ys,sz = getData(0)
        #ln, = plt.plot([], [], 'ok')
        #plt.xlim(-5,5)
        #plt.ylim(-5,5)
        plt.axis('equal')
        return scat,vplot,
    scat,vplot, = init()
    def update(frame):
        print "FRAME ", frame
        #xs,ys,sz = getData(frame)
        #ln.set_data(xs,ys)
        cpos,sz,vpos = getData(frame)
        scat .set_offsets(cpos)
        vplot.set_data(vpos[:,0],vpos[:,1])
        return scat,vplot,
    #ani = FuncAnimation(fig, update, frames=range(nframes), init_func=init, blit=True, interval=interval)
    ani = FuncAnimation(fig, update, frames=range(nframes), blit=True, interval=interval)
    return ani

if __name__ == "__main__":
    N    = 5
    nvs  = np.random.randint( 3,8, N, dtype=np.int32 );       print "nvs:   ", nvs
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
    print "vpos: ", vpos

    import matplotlib.pyplot as plt
    #plt.figure(); plotCycles(cpos=cpos)
    
    pcff.setupOpt(dt=0.1, damping=0.05, f_limit=1.0,v_limit=1.0 )
    #pcff.relaxNsteps(kind=0, nsteps=100)
    #for i in range(3000):
    #    pcff.relaxNsteps(kind=0, nsteps=1)
    #    #print cpos
    
    pcff.init(rots)
    
    Rs = 1/np.sin(np.pi/nvs)
    
    def getDate(frame):
        pcff.relaxNsteps(kind=1, nsteps=1)
        return cpos, Rs, vpos
    
    ani = make_anim(getDate, plt.gcf(), 10, interval=10 )
    print "vpos: ", vpos
    #print "vpos: ", vpos
    
    #plt.figure(); plotCycles(cpos=cpos,vpos=vpos,nvs=nvs)
    
    '''
    plt.plot(cpos[:,0],cpos[:,1],"+"); plt.axis("equal")
    plt.plot(vpos[:,0],vpos[:,1],".")
    iv=0
    for ni in nvs:
        vs=np.vstack( (vpos[iv:iv+ni,:],vpos[iv,:]) )
        plt.plot(vs[:,0],vs[:,1],".-")
        iv+=ni
    '''

    plt.show()
    #pcff.relaxNsteps(nsteps=10, F2conf=-1.0, dt=0.1, damp=0.9)








