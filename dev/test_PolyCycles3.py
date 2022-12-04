#!/usr/bin/python

import sys
import os
import numpy as np
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
import ppafm.atomicUtils as au
import ppafm.chemistry   as ch
import ppafm.MMFF as mmff

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

def scatterPoints2D( N, sz=2.5, Lrange=None ):
    if Lrange is None:
        Lrange=np.sqrt(N*sz*sz)
    ts=np.arange(N)/Lrange
    ring_pos = np.empty( (N,2) )
    ring_pos[:,0] = np.floor(ts)
    ring_pos[:,1] = (ts - ring_pos[:,0])*Lrange
    ring_pos[:,:] += (np.random.random(ring_pos.shape)-0.5)*0.5
    ring_pos*=2.5
    return ring_pos, Lrange

def generateRings( N=30, ngons=[5,6,7], ngon_rates=[0.2,0.6,0.2], sz=2.5 ):
    nvs=np.random.choice( ngons, size=N, p=ngon_rates )    #; print "nvs: " ,nvs
    ring_pos,Lrange = scatterPoints2D( N, sz=2.5 )
    ring_Rs = 0.5/np.sin(np.pi/nvs)
    # ------ Relax rings "
    opt = ch.FIRE()
    ch.relaxAtoms( ring_pos, ring_Rs, FFfunc=ch.getForceIvnR24, fConv=0.1, nMaxStep=1000, optimizer=opt )
    #ch.relaxAtoms( ring_pos, ring_Rs, FFfunc=ch.getForceIvnR24, fConv=0.0001, nMaxStep=1000, optimizer=opt )
    return ring_pos, ring_Rs, Lrange


def T1():
    global t1
    t1 = time.clock()

def T2(s):
    if bTime:
        print("Time spend in : "+s+" [s]: ", time.clock() - t1)

def generateAromaticMolecule(
        N=30, ngons=[5,6,7], ngon_rates=[0.2,0.6,0.2], sz=2.5, bondLenght=1.4,
        E12=0.5, E22=+0.5, E32=+0.5,
        ne2elem=[ 'C', 'N', 'O' ], epairProb=0.4, bNitrogenFix=True,
        fname="test_PolyCyclesMMFF.xyz", bMovie=False,
    ):
    T1();  ring_pos, ring_Rs, Lrange = generateRings( N=30, ngons=[5,6,7], ngon_rates=[0.2,0.6,0.2], sz=2.5 )                    ;T2("generateRings")
    T1();  atom_pos,bonds, atypes, nngs, neighs, ring_bonds, atom_N6mask = ch.ringsToMolecule( ring_pos, ring_Rs )               ;T2("ch.ringsToMolecule")
    T1();  bondOrder, atomOrder, _ = ch.estimateBondOrder( atypes, bonds, E12=0.5, E22=+0.5, E32=+0.5 )                          ;T2("ch.estimateBondOrder")

    T1();
    apos = (np.append(atom_pos, np.zeros((len(atom_pos),1)), axis=1) * bondLenght ).copy()
    npis, nepairs, elems = mmff.assignAtomTypes( atomOrder, nngs, epairProb=epairProb, bNitrogenFix=True, ne2elem=[ 'C', 'N', 'O' ] )
    apos,atypes,aelems   = mmff.relaxMolecule( apos, bonds, npis, nepairs, elems, fname=fname, bMovie=bMovie )
    T2("mmff.relaxMolecule")

    rings      = (ring_pos, ring_Rs, ring_bonds)
    mol2d      = (atom_pos,bonds)
    mol3d      = (apos,atypes,aelems)
    atom_conf1 = ( bondOrder, atomOrder, atom_N6mask )
    atom_conf2 = ( nngs, npis, nepairs )
    return mol3d,mol2d,rings,atom_conf1,atom_conf2

def plotRings( ring_pos, ring_Rs, ring_bonds=None ):
    plt.scatter(ring_pos[:,0], ring_pos[:,1], s=(ring_Rs*60)**2, c='none' )
    if ring_bonds is not None:
        for b in ring_bonds:
            pb=ring_pos[b,:]
            plt.plot( pb[:,0],pb[:,1],'-b' )

def plotMolecule( atom_pos, bonds=None, bondOrder=None, atom_N6mask=None ):
    plt.plot( atom_pos[:,0],atom_pos[:,1], "*" )
    if atom_N6mask is not None:
        plt.plot(atom_pos[atom_N6mask,0],atom_pos[atom_N6mask,1],"ow")
    if ring_bonds is not None:
        for i,b in enumerate(bonds):
            pb=atom_pos[b,:]
            if bondOrder is not None:
                plt.plot( pb[:,0],pb[:,1], '-r', lw=1+bondOrder[i]*5 )

if __name__ == "__main__":
    np.random.seed(26465)

    global bTime
    bTime = True



    t1 = time.clock()
    mol3d,mol2d,rings,atom_conf1,atom_conf2 = generateAromaticMolecule( )
    print("Time to generate molecule [s]: ", time.clock() - t1)

    (ring_pos, ring_Rs, ring_bonds) = rings
    (atom_pos,bonds)                      = mol2d
    (apos,atypes,aelems)                  = mol3d
    ( bondOrder, atomOrder, atom_N6mask ) = atom_conf1
    ( nngs, npis, nepairs )               = atom_conf2

    plt.figure()
    plotRings( ring_pos, ring_Rs, ring_bonds )
    plotMolecule( atom_pos, bonds, bondOrder, atom_N6mask )
    plt.axis('equal')
    plt.show()



    bTime = False

    for i in range(20):
        print("Generated molecule [%i] " %i)
        generateAromaticMolecule( fname="mol_out/mol_mmff_%04i.xyz" %i )
