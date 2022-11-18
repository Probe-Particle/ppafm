#!/usr/bin/python

import sys
import os
import numpy as np

from . import  RigidMol  as rmol
from . import basUtils

def combineGeoms(mol,surf):
    es   = mol[0] + surf[0]
    xyzs = np.hstack( [np.array( mol[1:4] ), np.array(  surf[1:4] )] ).transpose().copy()
    return es, xyzs

def sphereTangentSpace(n=100):
    golden_angle = np.pi * ( 3.0 - np.sqrt(5.0) )
    theta  = golden_angle * np.arange(n)
    z      = np.linspace(1.0 - 1.0/n, 1.0/n - 1.0, n)
    radius = np.sqrt( 1.0 - z*z )
    cas  = np.cos(theta)
    sas  = np.sin(theta)
    rots = np.zeros( (n,3,3) )
    rots[:,2,0] = radius * cas
    rots[:,2,1] = radius * sas
    rots[:,2,2] = z
    rots[:,0,0] = -sas
    rots[:,0,1] =  cas
    rots[:,1,:] =  np.cross( rots[:,2,:], rots[:,0,:] )
    return rots

def quat2mat(q):
    x=q[0]; y=q[1]; z=q[2]; w=q[3];
    r2 = x*x + y*y + z*z + w*w;
    s  = 2 / r2;
    xs = x * s;  ys = y * s;  zs = z * s;
    xx = x * xs; xy = x * ys; xz = x * zs;
    xw = w * xs; yy = y * ys; yz = y * zs;
    yw = w * ys; zz = z * zs; zw = w * zs;
    return np.array(  [
        [1 - (yy + zz),     (xy - zw),     (xz + yw) ],
        [    (xy + zw), 1 - (xx + zz),     (yz - xw) ],
        [    (xz - yw),     (yz + xw), 1 - (xx + yy) ]
    ] )

def mat2quat(m):
    t = m[0,0] + m[1,1] + m[2,2];
    if (t >= 0):
        s = np.sqrt(t + 1);
        w = 0.5 * s;
        s = 0.5 / s;
        x =  (m[2,1] - m[1,2]) * s;
        y =  (m[0,2] - m[2,0]) * s;
        z =  (m[1,0] - m[0,1]) * s;
    elif ((m[0,0] > m[1,1]) and (m[0,0] > m[2,2])):
        s = np.sqrt(1 + m[0,0] - m[1,1] - m[2,2]);
        x = s * 0.5;
        s = 0.5 / s;
        y = (m[1,0] + m[0,1]) * s;
        z = (m[0,2] + m[2,0]) * s;
        w = (m[2,1] - m[1,2]) * s;
    elif (m[1,1] > m[2,2]):
        s = np.sqrt(1 + m[1,1] - m[0,0] - m[2,2]);
        y = s * 0.5;
        s = 0.5 / s;
        x = (m[1,0] + m[0,1]) * s;
        z = (m[2,1] + m[1,2]) * s;
        w = (m[0,2] - m[2,0]) * s;
    else:
        s = np.sqrt(1 + m[2,2] - m[0,0] - m[1,1]);
        z = s * 0.5;
        s = 0.5 / s;
        x = (m[0,2] + m[2,0]) * s;
        y = (m[2,1] + m[1,2]) * s;
        w = (m[1,0] - m[0,1]) * s;
    return np.array([x,y,z,w])

def initSurf( surfFile, cell, ns=[60,60,100] ):
    rmol.initRigidSubstrate ( surfFile, np.array(ns,dtype=np.int32), np.array([0.0,0.0,0.0]), np.array(cell) )
    if os.path.isfile("data/FFPauli.bin"):
        print("gridFF found on disk => loading ")
        rmol.loadGridFF()
    else:
        print("gridFF not found on disk => recalc ")
        rmol.recalcGridFF( np.array([1,1,1],dtype=np.int32) )
        rmol.saveGridFF()

def getSurfConfs( rots, molFile, pos=[ 5.78, 6.7, 12.24 ], nMaxIter=200, Fconv=0.01 ):

    print("DEBUG 0")

    rmol.clear()

    mol   = rmol.loadMolType( molFile )                              ;print("DEBUG 0.1") 
    rot0  = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])    ;print("DEBUG 0.2")
    rmol.insertMolecule( mol, np.array(pos), rot0, True )          ;print("DEBUG 0.3")

    print("DEBUG 1")

    # ========= Relaxation

    rmol.bakeMMFF()
    rmol.prepareOpt()
    rmol.setOptFIRE( dt_max=0.2, dt_min=0.01, damp_max=0.1, minLastNeg=5, finc=1.1, fdec=0.5, falpha=0.98, kickStart=1.0 ); print("DEBUG 1.3")

    print("DEBUG 3")

    poses = rmol.getPoses()
    apos  = rmol.getAtomPos()

    rots_ = []
    for irot,rot in enumerate(rots):
        mol_name = molFile.split("/")[1].split(".")[0]
        print(mol_name)
        fname = "movie_%s_%03i.xyz" %(mol_name,irot)
        if os.path.exists(fname): os.remove(fname)
        q = mat2quat(rot)
        print("q ", q)
        poses[0,4:8] = q
        for i in range(nMaxIter):
            F2 = rmol.relaxNsteps( 1, 0.0 ); 
            rot_ = quat2mat(poses[0,4:8])
            rots_.append(rot_)
            xyzs[:nAtomMol,:] = apos[:,:]
            basUtils.saveXYZ(fname, xyzs, es, append=True)
        print("rot  ", rot)
        print("rot_ ", rot_)

    del  poses
    del  apos
    return rots_

if __name__ == "__main__":

    from . import basUtils as au

    os.chdir( "/u/25/prokoph1/unix/git/SimpleSimulationEngine/cpp/Build/apps/MolecularEditor2" )

    xyzs, Zs, qs, _ = au.loadXYZ("inputs/water_T5_ax.xyz")
    water = [list(Zs), list(xyzs[:, 0]), list(xyzs[:, 1]), list(xyzs[:, 2]), list(qs)]
    xyzs, Zs, qs, _ = au.loadXYZ("inputs/Campher.xyz")
    campher = [list(Zs), list(xyzs[:, 0]), list(xyzs[:, 1]), list(xyzs[:, 2]), list(qs)]
    xyzs, Zs, qs, _ = au.loadXYZ("inputs/Cu111_6x6_2L.xyz")
    surf = [list(Zs), list(xyzs[:, 0]), list(xyzs[:, 1]), list(xyzs[:, 2]), list(qs)]

    cell = [[15.31593,0.0,0.0],[0.0,13.26399,0.0],[0.0,0.0,20.0]]
    rots  = sphereTangentSpace(n=5)

    rmol.initParams( "common_resources/AtomTypes.dat", "common_resources/BondTypes.dat" )
    initSurf( "inputs/Cu111_6x6_2L.xyz", cell, ns=[60,60,100] )
    print("========== water_T5_ax.xyz ===========")
    print("========== water_T5_ax.xyz ===========")
    print("========== water_T5_ax.xyz ===========")
    nAtomMol = len(water[0])
    es, xyzs = combineGeoms(water,surf)
    rots_ = getSurfConfs( rots, "inputs/water_T5_ax.xyz", pos=[ 5.78, 6.7, 12.24 ],  nMaxIter=100, Fconv=0 )
    print("========== Campher.xyz ===========")
    print("========== Campher.xyz ===========")
    print("========== Campher.xyz ===========")
    nAtomMol = len(campher[0])
    es, xyzs = combineGeoms(campher,surf)
    rots_ = getSurfConfs( rots, "inputs/Campher.xyz", pos=[ 5.78, 6.7, 12.24 ], nMaxIter=100, Fconv=0 )
    print(">>>> ALL DONE <<<<")
