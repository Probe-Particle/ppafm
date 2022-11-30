#!/usr/bin/python

# https://matplotlib.org/examples/user_interfaces/index.html
# https://matplotlib.org/examples/user_interfaces/embedding_in_qt5.html
# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases


import sys
import os
import shutil
import time
import random
import matplotlib;
import numpy as np
from enum import Enum

import matplotlib as mpl;  mpl.use('Agg'); print("plot WITHOUT Xserver");
import matplotlib.pyplot as plt

#sys.path.append("/home/prokop/git/ProbeParticleModel_OCL") 
#import ppafm.GridUtils as GU

from   ppafm import basUtils
from   ppafm import PPPlot 
import ppafm.GridUtils as GU
import ppafm.common    as PPU
import ppafm.cpp_utils as cpp_utils

import pyopencl as cl
import ppafm.ocl.oclUtils as oclu
import ppafm.ocl.field    as FFcl
import ppafm.ocl.relax    as oclr

Modes     = Enum( 'Modes',    'LJel LJel_pbc LJQ' )
DataViews = Enum( 'DataViews','FFin FFout df FFel FFpl' )

FFcl.init()
oclr.init()

# ==== Setup

# --- Input files
dirNames  = ["out0"]
#dirNames  = ["out0", "out1", "out2", "out3", "out4", "out5", "out6" ] 
#geomFileNames = ["out0/pos.xyz", "out1/pos.xyz", "out2/pos.xyz", "out3/pos.xyz", "out4/pos.xyz", "out5/pos.xyz", "out6/pos.xyz" ]

# --- ForceField
mode = Modes.LJQ.name
pixPerAngstrome = 10
iZPP = 8
Q    = -0.1;
bPBC = True

# --- Relaxation

relax_dim   = ( 100, 100, 20)
lvec = np.array([
    [ 0.0,  0.0,  0.0],
    [19.0,  0.0,  0.0],
    [ 0.0, 20.0,  0.0],
    [ 0.0,  0.0, 21.0]
])

distAbove = 7.5
islices   = [0,+2,+4,+6,+8]

relax_params = np.array( [ 0.1,0.9,0.1*0.2,0.1*5.0], dtype=np.float32 );
dTip         = np.array( [ 0.0 , 0.0 , -0.1 , 0.0 ], dtype=np.float32 );
stiffness    = np.array( [0.24,0.24,0.0, 30.0     ], dtype=np.float32 ); stiffness/=-16.0217662;
dpos0    = np.array([0.0,0.0,0.0,4.0], dtype=np.float32 ); 
dpos0[2] = -np.sqrt( dpos0[3]**2 - dpos0[0]**2 + dpos0[1]**2 ); 
print("dpos0 ", dpos0)

# === Functions

def PBCAtoms3D( Zs, Rs, Qs, lvec, npbc=[1,1,1] ):
    '''
    multiply atoms of sample along supercell vectors
    the multiplied sample geometry is used for evaluation of forcefield in Periodic-boundary-Conditions ( PBC )
    '''
    Zs_ = []
    Rs_ = []
    Qs_ = []
    for iatom in range(len(Zs)):
        Zs_.append( Zs[iatom] )
        Rs_.append( Rs[iatom] )
        Qs_.append( Qs[iatom] )
    for ia in range(-npbc[0],npbc[0]+1):
        for ib in range(-npbc[1],npbc[1]+1):
            for ic in range(-npbc[2],npbc[2]+1):
                if (ia==0) and (ib==0) and (ic==0) :
                    continue
                for iatom in range(len(Zs)):
                    x = Rs[iatom][0] + ia*lvec[0][0] + ib*lvec[1][0] + ic*lvec[2][0]
                    y = Rs[iatom][1] + ia*lvec[0][1] + ib*lvec[1][1] + ic*lvec[2][1]
                    z = Rs[iatom][2] + ia*lvec[0][2] + ib*lvec[1][2] + ic*lvec[2][2]
                    #if (x>xmin) and (x<xmax) and (y>ymin) and (y<ymax):
                    Zs_.append( Zs[iatom] )
                    Rs_.append( (x,y,z)   )
                    Qs_.append( Qs[iatom] )
                    #print "i,j,iatom,len(Rs)", i,j,iatom,len(Rs_)
    return np.array(Zs_).copy(), np.array(Rs_).copy(), np.array(Qs_).copy()	

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis    =  np.asarray(axis)
    axis    =  axis/np.sqrt(np.dot(axis, axis))
    a       =  np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def genRotations( axis, thetas ):
    return np.array( [ rotation_matrix(axis, theta) for theta in thetas ] )

def maxAlongDir(atoms, hdir):
    #print atoms[:,:3]
    xdir = np.dot( atoms[:,:3], hdir[:,None] )
    #print xdir
    imin = np.argmax(xdir)
    return imin, xdir[imin][0]

#def getOptSlice( atoms, hdir ):
#    

def loadSpecies(fname):
    try:
        with open(fname, 'r') as f:  
            str_Species = f.read(); 
    except:
        print("defaul atomtypes.ini")
        with open(cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini', 'r') as f:  
            str_Species = f.read();
    str_Species = "\n".join( "\t".join( l.split()[:5] )  for l in str_Species.split('\n')  )
    print("str_Species")
    print(str_Species)
    return PPU.loadSpeciesLines( str_Species.split('\n') )

def evalFFatoms_LJC( atoms, cLJs, poss, func_runFF=FFcl.runLJC ):
    ff_args = FFcl.initArgsLJC( atoms, cLJs, poss )
    ff_nDim = poss.shape[:3]
    FF      = func_runFF( ff_args, ff_nDim )
    FFcl.releaseArgs(ff_args)
    return FF

def writeDebugXYZ( fname, lines, poss ):
    fout  = open(fname,"w")
    natom = int(lines[0])
    npos  = len(poss)
    fout.write( "%i\n" %(natom + npos) )
    fout.write( "\n" )
    for line in lines[2:natom+1]:
        fout.write( line )
    for pos in poss:
        fout.write( "He %f %f %f\n" %(pos[0], pos[1], pos[2]) )
    fout.write( "\n" )

# === Main

if __name__ == "__main__":
    rotations = genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, 8) )
    #rotations = genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, 1) )
    print("rotations: ", rotations)
    #exit()

    typeParams = loadSpecies('atomtypes.ini')
    #lvec       = np.genfromtxt('cel.lvs') 
    #lvec       = np.insert( lvec, 0, 0.0, axis=0); 
    print("lvec ", lvec)
    invCell = oclr.getInvCell(lvec)
    print("invCell ", invCell)
    ff_nDim       = np.array([
                int(round(pixPerAngstrome*(lvec[1][0]+lvec[1][1]))),
                int(round(pixPerAngstrome*(lvec[2][0]+lvec[2][1]))),
                int(round(pixPerAngstrome*(lvec[3][2]           )))
            ])
    print("ff_nDim ", ff_nDim)
    poss       = FFcl.getposs( lvec, ff_nDim )
    print("poss.shape", poss.shape)

    for dirName in dirNames:
        t1tot = time.clock()
        print(" ==================", dirName)
        t1prepare = time.clock();
        atom_lines =  open( dirName+"/pos.xyz" ).readlines()
        xyzs,Zs,enames,qs = basUtils.loadAtomsLines( atom_lines )
        natoms = len(Zs) 

        if(bPBC):
            #Zs, xyzs, qs = PPU.PBCAtoms( Zs, xyzs, qs, avec=self.lvec[1], bvec=self.lvec[2] )
            Zs, xyzs, qs = PBCAtoms3D( Zs, xyzs, qs, lvec[1:], npbc=[1,1,1] )
        atoms = FFcl.xyzq2float4(xyzs,qs)
        cLJs  = PPU.getAtomsLJ( iZPP, Zs, typeParams ).astype(np.float32)
        Tprepare = time.clock()-t1prepare;

        t1ff = time.clock();
        FF    = evalFFatoms_LJC( atoms, cLJs, poss, func_runFF=FFcl.runLJC )
        FEin =  FF[:,:,:,:4] + Q*FF[:,:,:,4:]

        Tff = time.clock()-t1ff;
        #GU.saveXSF( dirName+'/Fin_z.xsf',  FEin[:,:,:,2], lvec ); 

        print("FEin.shape ", FEin.shape);
        relax_args  = oclr.prepareBuffers( FEin, relax_dim )
        cl.enqueue_copy( oclr.oclu.queue, relax_args[0], FEin, origin=(0,0,0), region=FEin.shape[:3][::-1] )

        for irot,rot in enumerate(rotations):
            print("rotation #",irot, rot)
            subDirName = dirName + ("/rot%02i" %irot)
            if os.path.isdir(subDirName):
                #os.rm(subDirName)
                shutil.rmtree(subDirName)
            os.makedirs( subDirName )

            imax,xdirmax  = maxAlongDir(atoms[:natoms], rot[2] )
            print("imax ", imax, " xdirmax ", xdirmax)

            pos0 = atoms[imax,:3] + rot[2] * distAbove;
            dTip [:3] = rot[2]*-0.1
            dpos0[:3] = rot[2]*-dpos0[3]
            print("dots ", np.dot(rot[0],rot[1]), np.dot(rot[0],rot[2]), np.dot(rot[1],rot[2]))
            print("pos0", pos0, "\nrot", rot, "\ndTip", dTip, "\ndpos0", dpos0)
            relax_poss = oclr.preparePossRot( relax_dim, pos0, rot[0], rot[1], start=(-10.0,-10.0), end=(10.0,10.0) )
            print("relax_poss.shape ",relax_poss.shape)
            writeDebugXYZ( subDirName + "/debugPos0.xyz", atom_lines, relax_poss[::10,::10,:].reshape(-1,4) )
            #continue
            #exit()

            t1relax = time.clock();
            #region = FEin.shape[:3][::-1]
            #cl.enqueue_copy( oclr.oclu.queue, relax_args[0], FEin, origin=(0,0,0), region=region)
            #FEout = oclr.relax( relax_args, relax_dim, invCell, poss=relax_poss, FEin=FEin, dTip=dTip, dpos0=dpos0, stiffness=stiffness, relax_params=relax_params  )
            FEout = oclr.relax( relax_args, relax_dim, invCell, poss=relax_poss, dTip=dTip, dpos0=dpos0, stiffness=stiffness, relax_params=relax_params  )
            Trelax = time.clock() - t1relax;

            #writeDebugXYZ( subDirName + "/debugPosRlaxed.xyz", atom_lines, FEout[::10,::10,:,:].reshape(-1,4) )
            #GU.saveXSF( geomFileName+'_Fout_z.xsf',  FEout[:,:,:,2], lvec );
            #print "FEout.shape ", FEout.shape
            #np.save( subDirName+'/Fout_z.npy', FEout[:,:,:,2] )
            t1plot = time.clock();
            for isl in islices:
                plt.imshow( FEout[:,:,isl,2] )
                plt.savefig( subDirName+( "/FoutZ%03i.png" %isl ), bbox_inches="tight"  ); 
                plt.close()
            Tplot = time.clock()-t1plot;

            Ttot = time.clock()-t1tot;
            print("Timing[s] Ttot %f Tff %f Trelax %f Tprepare %f Tplot %f " %(Ttot, Tff, Trelax, Tprepare, Tplot))
        
        oclr.releaseArgs(relax_args)

#plt.show()

