#!/usr/bin/python

import sys
import os
import numpy as np
import time




#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import common    as PPU
#from optparse import OptionParser

import matplotlib.pyplot as plt

#from scipy.interpolate import CubicSpline

#import scipy.interpolate as interp #.CubicSpline as CS
#from scipy.interpolate import interp1d #.CubicSpline as CS
from scipy.interpolate import Akima1DInterpolator

xs = np.linspace(-1,4,200)
Xs = np.array([-1 , 0, 1,2, 3, 4])
Ys = np.array([20.,-1,-2,3,10,20])


Es = {
"C1": [20.,0,0, 1,10,20],
"C2": [20.,0,0, 2,10,20],
"C3": [20.,0,-1,3,10,20],
}

typeEs = [
 [20.,0,0, 1,10,20],
 [20.,0,0, 2,10,20],
 [20.,0,-1,3,10,20],
]

'''
colors=["k","b","r"]


def plotInterp(Xs,Ys,label,c):
    #sp = interp.CubicSpline( xs, ys )
    #f = interp1d(xs,ys)
    f  = Akima1DInterpolator(Xs,Ys)
    df = f.derivative()
    ys  = f (xs)
    dys = df(xs)
    plt.subplot(2,1,1); plt.axhline(0,c="k",ls='--')
    plt.plot(xs,ys,"-",c=c)
    plt.plot(Xs,Ys,"+",c=c)
    plt.subplot(2,1,2); plt.axhline(0,c="k",ls='--')
    plt.plot(xs,dys,"-",c=c,label=label)

plt.figure(figsize=(5,10))

i=0
for name,Ys in Es.iteritems():
    plotInterp(Xs,Ys,name,colors[i])
    i+=1

#ys = (xs-np.round(xs))**2 * 10
ys = np.cos(2*np.pi*xs) * -10

plt.legend(loc=4)

plt.subplot(2,1,1);
plt.plot(xs,ys,"-",c="m", label="BO")


plt.show()
#exit()
'''




import pyProbeParticle.PolyCycles  as pcff
import pyProbeParticle.atomicUtils as au
import pyProbeParticle.chemistry   as ch






'''
species = [
[("-CH3" ,1), ("=CH2",1),("=NH" ,1),("-NH2",1),("-OH",1),("=O",1),("-Cl",1) ],  # 1 bond
[("-CH2-",1), ("=CH-",1),("-NH-",1),("=NH-",1),("-O-" ,1) ],                    # 2 bond
[("C",1),("N",1)],                                                              # 3 bond
]
'''

'''
species = [
[("C",1),("N",1),("N",1),("O",1),("Cl",1) ],  # 1 bond
[("C",1),("N",1),("N",1),("O",1) ],                    # 2 bond
[("C",1),("N",1)],                                                              # 3 bond
]
'''

species = [
[("C",2),("N",1),("O",1),("F",1),("Cl",1), ],  # 1 bond
[("C",4),("N",1),("O",1) ],                    # 2 bond
[("C",8),("N",0.5)],                                                              # 3 bond
]


'''
species = [
[("F",1),("Cl",1),("Br",1),("I",1) ],  # 1 bond
[("O",1),("S",1),("Se",1) ],                    # 2 bond
[("C",1),("N",1)],                                                              # 3 bond
]
'''

'''
species = [
[("Cl",1),], # 1 bond
[("O",1),],  # 2 bond
[("C",1),]   # 3 bond
]
'''

#species = normalizeSpeciesProbs(species)
plevels = ch.speciesToPLevels(species)
print "plevels", plevels
#exit()

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

    #plt.figure(); plotCycles(cpos=cpos)
    
    pcff.setupOpt(dt=0.3, damping=0.05, f_limit=1.0,v_limit=1.0 )
    pcff.relaxNsteps(kind=0, nsteps=1000)

    #pcff.init(rots)
    
    Rs = 0.5/np.sin(np.pi/nvs)
    plt.scatter(cpos[:,0], cpos[:,1], s=(Rs*60)**2, c='none' )
    #print "nvs ",nvs
    #print "Rs ",Rs
    
    bonds  = ch.findBonds(cpos,Rs,fR=1.2)
    neighs = ch.bonds2neighs(bonds,N)
    #print "neighs ", neighs

    tris,tbonds = ch.findTris_(bonds,neighs)
    #print "tris  ",tris
    #print "tbonds   ",tbonds 
    
    tbonds_,_ = ch.tris2num_(tris, tbonds)
    #print "tbonds_  ",tbonds_ 
    
    tneighs = ch.bonds2neighs( tbonds_, len(tris) )
    print tneighs
    
    #tris_ = ch.tris2num(tris)
    #print "tris_ ", tris_
    
    ops = ch.trisToPoints(tris,cpos)
    #print "ops", ops
    plt.plot( ops[:,0],ops[:,1], "*" )
    
    '''
    fout = open("test_PolyCycles.xyz","w")
    n=len(tneighs)
    fout.write("%i \n" %n )
    fout.write("#comment \n" )
    enames = ["Cl","O","C"]
    fbl = 1.3
    for i in range(n):
        nng = len(tneighs[i])-1
        ename=enames[nng]
        fout.write("%s %f %f %f \n" %(ename,ops[i,0]*fbl,ops[i,1]*fbl,0) )
    fout.close()
    '''
    
    #elem_names = [ s[0]     for s   in species ]
    nngs       = [ len(ngs) for ngs in tneighs ] 
    elist = ch.selectRandomElements( nngs, species, plevels )
    print "elist", elist
    
    #print ops.shape, np.zeros(len(ops)).shape
    xyzs = np.append(ops, np.zeros((len(ops),1)), axis=1)
    #print "xyzs", xyzs
    au.saveXYZ( elist,xyzs*1.3, "test_PolyCycles.xyz" )
    
    
    bo,ao = ch.relaxBondOrder( nngs, tbonds_, typeEs, Nstep=100, EboMax=100 )
    
    for b in tbonds_:
        pb=ops[b,:]
        plt.plot( pb[:,0],pb[:,1],'-r' )
    
    #exit()
    for b in bonds:
        pb=cpos[b,:]
        plt.plot( pb[:,0],pb[:,1],'-b' )
    plt.axis('equal')
    plt.show()
    #pcff.relaxNsteps(nsteps=10, F2conf=-1.0, dt=0.1, damp=0.9)








