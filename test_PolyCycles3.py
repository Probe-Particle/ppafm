#!/usr/bin/python

import sys
import os
import numpy as np
import time

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import common    as PPU
#from optparse import OptionParser

#import pyProbeParticle.PolyCycles  as pcff
import pyProbeParticle.atomicUtils as au
import pyProbeParticle.chemistry   as ch

import matplotlib.pyplot as plt

import pyProbeParticle.MMFF as mmff

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

Es = {
"C1": [20.,1,-2,1,10,20],
"C2": [20.,1,-2,2,10,20],
"C3": [20.,1,-2,3,10,20],
}

typeEs = [
 [20.,0,0, 1,10,20],
 [20.,0,0, 2,10,20],
 [20.,0,-1,3,10,20],
]

typeEs = [
 [20.,1,-1,1,10,20],
 [20.,1,-1,2,10,20],
 [20.,1,-1,3,10,20],
]


species = [
[("C",2),("N",1),("O",1),("F",1),("Cl",1), ],  # 1 bond
[("C",4),("N",1),("O",1) ],                    # 2 bond
[("C",8),("N",0.5)],                                                              # 3 bond
]


groupDict   = {
#  an,ao
 ( 1,0 ): [ ("-CH3" ,1),("-NH2",1),("-OH",1),("-F"   ,1),("-Cl",1) ],
 ( 1,1 ): [ ("=CH2" ,1),("=NH" ,1),("=O" ,1)                       ],
 ( 1,2 ): [ ("#CH"  ,1),("#N"  ,1)                                 ],
 ( 2,0 ): [ ("-CH2-",1),("-NH-",1),("-O-",1)                       ],
 ( 2,1 ): [ ("=CH-" ,1),("=N-" ,1)                                 ],
 ( 3,0 ): [ ("*CH"  ,1),("*N"  ,1)                                 ],
 ( 3,1 ): [ ("*C"   ,1)                                            ],
}


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
    np.random.seed(26465)
    
    print "#======== Generate Random Rings "
    Nring    = 30
    #nvs  = np.random.randint( 5,8, Nring, dtype=np.int32 );       #print "nvs:   ", nvs
    nvs=np.random.choice([5,6,7],size=Nring,p=[0.2,0.6,0.2] )
    #nvs=np.random.choice([4,5,6,7],size=Nring,p=[0.05,0.2,0.65,0.1] )
    #nvs = np.ones()*6
    print "nvs: " ,nvs
    
    #nv = pcff.setup( np.array(nvs,np.int32) )
    #ring_pos,vpos=pcff.getPos(Nring,nv)
    #ring_pos[:,0]=np.arange(Nring)*2.5
    #ring_pos[:,1]=np.sin(ring_pos[:,0]*0.3)
    
    #ring_pos[:,0]=np.arange(Nring)*2.5
    #ring_pos[:,1]=np.fract(ring_pos[:,0]/np.sqrt(Nring))
    
    ring_pos = np.empty( (Nring,2) )
    
    L=np.sqrt(Nring)
    ts=np.arange(Nring)/L
    ring_pos[:,0] = np.floor(ts)
    ring_pos[:,1] = (ts - ring_pos[:,0])*L
    ring_pos[:,:] += (np.random.random(ring_pos.shape)-0.5)*0.5
    ring_pos*=2.5
    
    #ring_pos[:,0],ring_pos[:,1]=np.modf( np.arange(Nring)*np.sqrt(Nring))
    
    #print "ring_pos: ", ring_pos
    
    ring_Rs = 0.5/np.sin(np.pi/nvs)
    
    #ring_pos_bak = ring_pos.copy()
    #plt.show()
    
    #pcff.setupOpt(dt=0.3, damping=0.05, f_limit=1.0,v_limit=1.0 )
    #pcff.relaxNsteps(kind=0, nsteps=1000)
    
    print "# ------ Relax rings "
    opt = ch.FIRE()
    ch.relaxAtoms( ring_pos, ring_Rs, FFfunc=ch.getForceIvnR24, fConv=0.1, nMaxStep=1000, optimizer=opt )
    #ch.relaxAtoms( ring_pos, ring_Rs, FFfunc=ch.getForceIvnR24, fConv=0.0001, nMaxStep=1000, optimizer=opt )

    cog = np.sum(ring_pos,axis=0)/Nring


    print "#========== Rings to atoms & bonds "
    
    ring_bonds  = ch.findBonds(ring_pos,ring_Rs,fR=1.0)
    ring_neighs = ch.bonds2neighs(ring_bonds,Nring)
    ring_nngs   = np.array([ len(ng) for ng in ring_neighs ],dtype=np.int)

    tris,bonds_ = ch.findTris(ring_bonds,ring_neighs)
    atom2ring   = np.array( list(tris), dtype=np.int )
    #print "atom2ring ", atom2ring

    atom_pos = ( ring_pos[atom2ring[:,0]] + ring_pos[atom2ring[:,1]] + ring_pos[atom2ring[:,2]] )/3.0
    #ops = ch.trisToPoints(tris,ring_pos)
    #print "ops", ops
    bonds,_ = ch.tris2num_(tris, bonds_)
    
    print "# ------ remove some atoms "
    print "pre:len ", len(atom_pos)
    mask    = ch.removeBorderAtoms(atom_pos,cog,L)
    print "remove atom mask ", mask
    bonds   = ch.validBonds( bonds, mask, len(atom_pos) )
    print "rm:bonds ", bonds
    atom2ring = atom2ring[mask,:]
    #print "rm:atom2ring ", atom2ring
    atom_pos  = atom_pos [mask,:]
    #print "rm:atom_pos ", atom_pos
    print "rm:len ", len(atom_pos)
    
    print "# ----- Hex mask "
    #N6mask = (ring_natm[:]==ring_nngs[:])
    ring_natm   = ch.getRingNatom(atom2ring,len(ring_neighs))
    ring_N6mask = np.logical_and( ring_natm[:]==6, ring_nngs[:]==6 )
    atom_N6mask = np.logical_or( ring_N6mask[atom2ring[:,0]], 
                  np.logical_or( ring_N6mask[atom2ring[:,1]], 
                                 ring_N6mask[atom2ring[:,2]]  ) )
    
    print "ring_natm ", ring_natm
    print "ring_nngs ", ring_nngs
    #print "N6mask    ", N6mask   #;exit()
    print "ring_N6mask", len(ring_N6mask), ring_N6mask
    print "atom_N6mask", len(atom_N6mask), atom_N6mask

    bonds = np.array(bonds)
    #print "tbonds_  ",tbonds_ 
    
    neighs  = ch.bonds2neighs( bonds, len(atom_pos) )
    print neighs
    
    #tris_ = ch.tris2num(tris)
    #print "tris_ ", tris_
    
    nngs  = np.array([ len(ngs) for ngs in neighs ],dtype=np.int) 
    
    #atypes=np.zeros(len(nngs),dtype=np.int)
    atypes=nngs.copy()-1
    #atypes[atom_N6mask]=0
    atypes[atom_N6mask]=3
    print "atypes ", atypes
    
    '''
    Eb2=+0.1
    typeEs = np.array([
     [20.,0,Eb2,0,10 ,20], # nng=1
     [20.,0,Eb2,2,10 ,20], # nng=2
     [20.,0,Eb2,3,10 ,20], # nng=3
     [20.,1, -4,3,10 ,20], # hex
    ])
    '''
    
    print " ================= Rleax BO "
    
    typeEs = ch.simpleAOEnergies( E12=0.5, E22=+0.5, E32=+0.5 )
    
    typeMasks, typeFFs = ch.assignAtomBOFF(atypes, typeEs)
    opt = ch.FIRE(dt_max=0.1,damp_max=0.25)
    
    bo,ao = ch.relaxBondOrder( bonds, typeMasks, typeFFs, fConv= 0.0001         , optimizer=opt, EboStart=0.0,  EboEnd=0.0                )
    #opt.bFIRE=False; opt.damp_max = 0.1
    bo,ao = ch.relaxBondOrder( bonds, typeMasks, typeFFs, fConv=-1., nMaxStep=100, optimizer=opt, EboStart=0.0,  EboEnd=10.0 , boStart=bo  )
    bo,ao = ch.relaxBondOrder( bonds, typeMasks, typeFFs, fConv=0.0001         , optimizer=opt, EboStart=10.0, EboEnd=10.0 , boStart=bo )
    
    #bo,ao = ch.relaxBondOrder( bonds, typeMasks, typeFFs, Nstep=50, dt=0.1, EboStart=0.0,  EboEnd=0.0,            )
    #bo,ao = ch.relaxBondOrder( bonds, typeMasks, typeFFs, Nstep=50, dt=0.1, EboStart=0.0,  EboEnd=10.0,boStart=bo )
    #bo,ao = ch.relaxBondOrder( bonds, typeMasks, typeFFs, Nstep=20, dt=0.1, EboStart=10.0, EboEnd=10.0,boStart=bo )
    
    print "ao ", ao[:6]
    
    print " ======== to xyz "
    
    '''
    # --- simple elements
    xyzs = np.append(atom_pos, np.zeros((len(atom_pos),1)), axis=1)*1.4
    
    #elem_names = [ s[0]     for s   in species ]
    #elist = ch.selectRandomElements( nngs, species, plevels )
    #print "elist", elist
    #print "xyzs", xyzs
    #au.saveXYZ( elist,xyzs, "test_PolyCycles.xyz" )
    
    # --- groups -> atoms
    groupDict = ch.makeGroupLevels(groupDict)
    aoi       = np.round(ao).astype(np.int)
    groups    = ch.selectRandomGroups( nngs, aoi, groupDict )
    
    #for i in range(len(groups)):
    #    if(groups[i]=="-NH2"):
    #        groups[i]="#CH"
    
    print "groups ", groups
    xyzs_g, elems_g = ch.groups2atoms( groups, neighs, xyzs )
    print "elems ", elems_g
    print "elems ", xyzs_g
    au.saveXYZ( elems_g,xyzs_g, "test_PolyCycles_g.xyz" )
    '''
    
    print " ======== MMFF relaxation "
    
    apos  = (np.append(atom_pos, np.zeros((len(atom_pos),1)), axis=1) * 1.4 ).copy()
    na    = len(apos); print "natom ",na  
    aconf = np.zeros( (len(apos),2), dtype=np.int32 )
    bonds = bonds.astype(np.int32).copy()

    nng = np.zeros(len(apos), dtype=np.int)
    for b in bonds: 
        nng[b[0]]+=1; nng[b[1]]+=1;
    print "nng", nng
    for i in range(len(nng)): print i,nng[i]


    aoi       = np.round(ao).astype(np.int32).copy()
    epairProb = 0.4
    nemax  = (4-nngs-aoi)
    ne_rnd =  epairProb * np.random.rand(na)
    print " ne_rnd ", ne_rnd
    
    aconf[:,0] = aoi                                                   # pi
    aconf[:,1] = np.round(  nemax*ne_rnd ) # epairs
    
    print "aoi   ", aoi
    print "nngs  ", nngs
    print "ne_max", nemax
    
    #            0    1    2
    ne2elem = [ 'C', 'N', 'O' ]
    elems = [   ne2elem[ne] for ne in aconf[:,1] ]
    print "elems ", elems
    
    print "aconf \n", aconf.T
    
    mmff.addAtoms(apos, aconf )
    mmff.addBonds(bonds, l0s=None, ks=None)   ;print "DEBUG 2"

    natom = mmff.buildFF(True,True,True)           ;print "DEBUG 4"
    #natom = mmff.buildFF(False,True,True)           ;print "DEBUG 4"
    mmff.setNonBonded(None)                        ;print "DEBUG 3"   # this activate NBFF with default atom params
    mmff.setupOpt()                                ;print "DEBUG 5"
    pos = mmff.getPos(natom)

    def runMMFF( Fconv=1e-6, Nmax=1000, perSave=10 ):
        pos = mmff.getPos(natom)
        for i in xrange(Nmax/perSave):
            f = mmff.relaxNsteps(perSave, Fconv=Fconv)
            if(f<1e-6): raise StopIteration
            yield pos,f

    elems += ['H']*(natom-len(elems))

    '''
    perSave=10
    errs = []
    fout = open("test_MMFF_movie.xyz", "w")
    for i in range(1000):
        print i," ",
        errs.append( f )
        au.writeToXYZ( fout, es, pos )
    fout.close()
    '''
    
    with open("test_PolyCyclesMMFF_movie.xyz", "w") as fout:
        for i,(pos,f) in enumerate(runMMFF()):
            #print i, f, pos
            print ">> save xyz", i, f
            au.writeToXYZ( fout, elems, pos )

    mmff.relaxNsteps(100)
    au.saveXYZ( elems, apos, "test_PolyCyclesMMFF.xyz", qs=None )
    
    exit()
    
    print " ======== Ploting "
    
    plt.figure()
    plt.scatter(ring_pos[:,0], ring_pos[:,1], s=(ring_Rs*60)**2, c='none' )
    plt.plot( atom_pos[:,0],atom_pos[:,1], "*" )
    
    for i,b in enumerate(bonds):
        pb=atom_pos[b,:]
        plt.plot( pb[:,0],pb[:,1], '-r', lw=1+bo[i]*5 )

    #exit()
    for b in ring_bonds:
        pb=ring_pos[b,:]
        plt.plot( pb[:,0],pb[:,1],'-b' )
    
    print atom_pos.shape, atom_N6mask.shape
    plt.plot(atom_pos[atom_N6mask,0],atom_pos[atom_N6mask,1],"ow")
    
    
    plt.axis('equal')
    plt.show()
    #pcff.relaxNsteps(nsteps=10, F2conf=-1.0, dt=0.1, damp=0.9)








