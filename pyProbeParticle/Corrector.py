
import sys
import os
import shutil
import time
import random
import matplotlib
import numpy as np
from enum import Enum

#import matplotlib.pyplot as plt

import pyopencl     as cl

from . import atomicUtils as au
from . import basUtils
from . import common    as PPU
from . import elements
#from . import oclUtils     as oclu
#from . import fieldOCL     as FFcl
#from . import RelaxOpenCL  as oclr
#from . import HighLevelOCL as hl

#from . import AFMulatorOCL_Simple
#from . import AuxMap
#from . import FARFF            #as Relaxer
#from . import GeneratorOCL_Simple2

# ======================================================
# ================== Class  Molecule
# ======================================================

class Molecule():
    """
    A simple class for Molecule data containing atom xyz positions, element types and charges.
    Arguments:
        xyzs: list, xyz positions of atoms in the molecule
        Zs: list, element types
        qs: list, charges
    """

    def __init__(self, xyzs, Zs, qs ):
        self.xyzs = xyzs
        self.Zs   = Zs
        self.qs   = qs 

    def clone(self):
        xyzs = self.xyzs
        Zs   = self.Zs
        qs   = self.qs
        return Molecule(xyzs,Zs,qs)

    def __str__(self):
        return np.c_[self.xyzs, self.Zs, self.qs].__str__()

    def __len__(self):
        return len(self.xyzs)

    def cog(self):
        n = len(self.xyzs)
        x = self.xyzs[:,0].sum()
        y = self.xyzs[:,1].sum()
        z = self.xyzs[:,2].sum()
        return np.array( [x,y,z] )/n

    def toXYZ( self, xyzfile, comment="#comment" ):
        au.writeToXYZ( xyzfile, self.Zs, self.xyzs, qs=self.qs, commet=comment )

# ======================================================
# ================== free function operating on Moleule
# ======================================================

def removeAtoms( molecule, nmax=1 ):
    #print( "----- removeAtoms  p0 ", p0 )
    xyzs = molecule.xyzs
    Zs   = molecule.Zs
    qs   = molecule.qs

    nmax = min(nmax, xyzs.shape[0])
    rem_idx = np.random.randint(nmax+1)
    sel = np.random.choice(np.arange(xyzs.shape[0]), rem_idx, replace=False)

    xyzs_ = np.delete( xyzs, sel, axis=0 )
    Zs_   = np.delete( Zs,   sel )
    qs_   = np.delete( qs,   sel )
    return Molecule(xyzs_, Zs_, qs_), sel

def addAtom( molecule, p0, R=0.0, Z0=1, q0=0.0, dq=0.0 ):
    # ToDo : it would be nice to add atoms in a more physicaly reasonable way - not overlaping, proper bond-order etc.
    # ToDo : currently we add always hydrogen - should we somewhere randomly pick different atom types ?
    xyzs = molecule.xyzs
    Zs   = molecule.Zs
    qs   = molecule.qs
    #print( "----- addAtom  p0 ", p0 )
    #dp    = (np.random.rand(3)-0.5)*R
    #dp[2] = 0
    min_x = xyzs.min(axis=0)
    max_x = xyzs.max(axis=0)
    min_x[2] = max_x[2]-1.0
    nx = np.random.uniform(min_x, max_x)[np.newaxis, ...]

    min_q = qs.min(axis=0)
    max_q = qs.max(axis=0)
    nq = np.random.uniform(min_q, max_q, (1,))

    Zs_ = np.append(Zs, np.array([Z0,]), axis=0)
    xyzs_ = np.append(xyzs, nx, axis=0 )
    qs_   = np.append(qs,   nq, axis=0 )
    return Molecule(xyzs_, Zs_, qs_)

def moveAtom( molecule, ia, dpMax=np.array([1.0,1.0,0.25]) ):
    xyzs         = molecule.xyzs.copy()
    Zs           = molecule.Zs.copy()
    qs           = molecule.qs.copy()
    xyzs[ia,:] += (np.random.rand(3)-0.5)*dpMax
    return Molecule(xyzs, Zs, qs)

def moveAtoms( molecule, p0, R=0.0, dpMax=np.array([1.0,1.0,0.25]), nmax=1 ):
    # ToDo : it would be nice to add atoms in a more physicaly reasonable way - not overlaping, proper bond-order etc.
    # ToDo : currently we add always hydrogen - should we somewhere randomly pick different atom types ?
    xyzs         = molecule.xyzs.copy()
    Zs           = molecule.Zs.copy()
    qs           = molecule.qs.copy()
    # --- choose nmax closest atoms
    rs           = (xyzs[:,0]-p0[0])**2 + (xyzs[:,1]-p0[1])**2
    sel          = rs.argsort()[:nmax] # [::-1]
    # --- move them randomly
    xyzs[sel,:] += (np.random.rand(len(sel),3)-0.5)*dpMax[None,:]
    return Molecule(xyzs, Zs, qs)

def moveMultipleAtoms(molecule: Molecule, scale=0.5, nmax=10):
    """
    A function for slightly shifting multiple atoms in the molecule.
    """
    nmax = nmax if nmax < len(molecule) else len(molecule)

    move_n = np.random.randint(nmax+1)
    to_be_moved = np.random.choice(len(molecule), move_n, replace=False)
    magnitudes = np.random.uniform(-scale, scale, (move_n, 3))

    xyzs_ = molecule.xyzs.copy()
    Zs_   = molecule.Zs.copy()
    qs_   = molecule.qs.copy()

    xyzs_[to_be_moved] = xyzs_[to_be_moved] + magnitudes

    return Molecule(xyzs_, Zs_, qs_)

def paretoNorm( Err1, Err2 ):
    '''
    this norm evaluate ares where Error got better (sum1), and where error got worse (sum2)
    we only accept moves where sum1>0 and sum2~0
    motivation is to make sure our move does not make things worse
    '''
    diff = Err1 - Err2
    Ebetter =  diff[diff>0].sum()   # got better
    Eworse  = -diff[diff<0].sum()   # got worse
    return Ebetter, Eworse

def paretoNorm_( Err1, Err2, trash=0.0000001 ):
    '''
    this norm evaluate ares where Error got better (sum1), and where error got worse (sum2)
    we only accept moves where sum1>0 and sum2~0
    motivation is to make sure our move does not make things worse
    '''
    diff = Err1 - Err2      # positive diff is improvement
    Ebetter =  diff-trash; Ebetter[Ebetter<0]=0
    Eworse  = -diff-trash; Eworse [Eworse <0]=0
    return Ebetter, Eworse


def blur( F ):
    #print( "F.shape ", F.shape )
    return ( F[:-1,:-1,:] + F[1:,:-1,:] + F[:-1,1:,:] + F[1:,1:,:] )*0.25

# ======================================================
# ================== Class  Corrector
# ======================================================

class Corrector():

    def __init__(self ):
        self.izPlot     = -8
        self.logImgName = None
        self.xyzLogFile = None

        self.best_E    = None
        self.best_mol  = None
        self.best_diff = None
        self.dpMax=np.array([0.5,0.5,0.15])

    def modifyStructure(self, molIn ):
        #mol = molIn.clone()
        #molOut.xyzs[0,0] += 0.1
        #molOut.xyzs[1,0] -= 0.1
        #p0  = (np.random.rand(3) - 0.5)
        #p0 += molIn.cog()
        #p0[0:2] *= 20.0 # scale x,y
        #p0[  2] *= 1.0  # scale z
        #R  = 3.0
        #ia = np.random.randint( 0 , len(molIn.Zs) )   # pick random atom
        # atoms 24 and 11 has moved
        #ia = 12   # pick Atom by hand  # Neighbor atom
        #ia = 23    # pick Atom by hand  # correct atom
        #ia = 15    # pick Atom by hand  # Always Worse
        sel=[24,11]; ipick=np.random.randint(0,2)
        ia = sel[ipick]-1
        molOut = moveAtom( molIn, ia, dpMax=self.dpMax )   # ToDo : This is just token example - later need more sophisticated Correction strategy
        # --- ToDo: Relaxation Should be possible part of relaxation ????
        return molOut

    def debug_plot(self, itr, AFM_Err, AFMs, AFMRef, Err ):
        if self.logImgName is not None:
            plt = self.plt
            plt.figure(figsize=(5*3,5))
            vmax=AFMRef[:,:,self.izPlot].max()
            vmin=AFMRef[:,:,self.izPlot].min()
            v2max=np.maximum(vmin**2,vmax**2)
            plt.subplot(1,3,1); plt.imshow( AFMRef[:,:,self.izPlot] ); #plt.title("AFMref" ); plt.grid()
            plt.subplot(1,3,2); plt.imshow( AFMs  [:,:,self.izPlot], vmin=vmin, vmax=vmax ); #plt.title("AFM[]"  ); plt.grid()
            plt.subplot(1,3,3); plt.imshow( np.sqrt(AFM_Err[:,:,self.izPlot]), vmin=0, vmax=np.sqrt(v2max)*0.2 ); #plt.title("AFMdiff"); plt.grid()
            plt.title( "Error=%g" %Err )
            plt.savefig( self.logImgName+("_%03i.png" %itr), bbox_inches='tight')
            plt.close  ()

    def debug_plot_Worse(self, itr, AFMs, AFMRef, ErrB, ErrW ):
        if self.logImgName is not None:
            plt = self.plt
            plt.figure(figsize=(5*4,5))
            vmax=AFMRef[:,:,self.izPlot].max()
            vmin=AFMRef[:,:,self.izPlot].min()
            v2max=np.maximum(vmin**2,vmax**2)
            plt.subplot(1,4,1); plt.imshow( AFMRef [:,:,self.izPlot], origin='image' ); #plt.title("AFMref" ); plt.grid()
            plt.subplot(1,4,2); plt.imshow( AFMs   [:,:,self.izPlot], origin='image', vmin=vmin, vmax=vmax ); #plt.title("AFM[]"  ); plt.grid()
            plt.subplot(1,4,3); plt.imshow( np.sqrt(ErrB[:,:,self.izPlot]), origin='image', vmin=0, vmax=np.sqrt(v2max)*0.2 ); plt.title("ErrB %g "%(ErrB.sum())  );
            plt.subplot(1,4,4); plt.imshow( np.sqrt(ErrW[:,:,self.izPlot]), origin='image', vmin=0, vmax=np.sqrt(v2max)*0.2 ); plt.title("ErrW %g "%(ErrW.sum())  );
            #plt.title( "Error=%g" %Err )
            plt.savefig( self.logImgName+("_%03i.png" %itr), bbox_inches='tight')
            plt.close  ()



    def try_improve(self, molIn, AFMs, AFMRef, itr=0 ):
        #print( " AFMs ", AFMs.shape, " AFMRef ", AFMRef.shape )
        AFMdiff = AFMs - AFMRef
        AFMdiff = blur( AFMdiff ); AFMdiff = blur( AFMdiff ) # BLUR
        AFMdiff2=AFMdiff**2 
        Err  = np.sqrt( AFMdiff2.sum() )  # root mean square error
        # ToDo : identify are of most difference and make random changes in that area
        #self.debug_plot( itr, AFMdiff2, AFMs, AFMRef, Err )
        Eworse = 0
        bBetter = False
        if( self.best_E is not None ):
            #ErrB,ErrW = paretoNorm_( self.best_diff, AFMdiff2 ); Eworse = ErrW.sum(); 
            #self.debug_plot_Worse( itr, AFMs, AFMRef, ErrB, ErrW )
            if( self.best_E > Err ):
                #Ebetter,Eworse = paretoNorm( self.best_diff, AFMdiff2 )
                ErrB,ErrW = paretoNorm_( self.best_diff, AFMdiff2 ); Eworse = ErrW.sum(); 
                ErrPar = Err + 10.*Eworse
                #ErrPar = Err + 0.*Eworse
                print( "\nmaybe better ? ", self.best_E , " <? ", ErrPar, " Eworse ", Eworse  )
                #self.debug_plot_Worse( itr, AFMs, AFMRef, ErrB, ErrW )
                #self.debug_plot( itr, AFMdiff2, AFMs, AFMRef, Err )
                #molIn.toXYZ( self.xyzLogFile, comment=("Corrector [%i] Err %g " %(itr,Err) ) )
                if ( self.best_E > ErrPar ):  # check if some areas does not got worse
                    bBetter = True
                    print( "[%i]SUCCESS : Err:" %itr, Err, " best: ", self.best_E, " Eworse ", Eworse  )
            if not bBetter:
                print("*",end='',flush=True)
                #print( "[%i]Err:" %itr, Err, " best: ", self.best_E  )

        if ( self.best_E is None ) or bBetter:
            self.debug_plot( itr, AFMdiff2, AFMs, AFMRef, Err )
            self.best_mol  = molIn 
            self.best_E    = Err
            self.best_diff = AFMdiff2
            if self.xyzLogFile is not None:
                self.best_mol.toXYZ( self.xyzLogFile, comment=("Corrector [%i] Err %g " %(itr,self.best_E) ) )
        #print( "Corrector.try_improve Err2 ", Err  )
        molOut = self.modifyStructure( self.best_mol )
        return Err, molOut

# ======================================================
# ================== Class  Mutator
# ======================================================

class Mutator():
    """
    A class for creating simple mutations to a molecule. The current state of the class allows for adding,
    removing and moving atoms. The mutations are not necessarily physically sensible at this point.
    Parameters:
        maxMutations: Integer. Maximum number of mutations per mutant. The number of mutations is chosen
        randomly between [1, maxMutations].
    """
    # Strtegies contain
    strategies = [
        (0.1,removeAtoms,{}),
        (0.0,addAtom,{}),
        (0.0,moveAtom,{})
    ]

    def __init__(self, maxMutations=10):
        self.setStrategies()
        self.maxMutations = maxMutations

    def setStrategies(self, strategies=None):
        if strategies is not None: self.strategies = strategies
        self.cumProbs = np.cumsum( [ it[0] for it in self.strategies ] )
        #print( self.cumProbs ); exit()

    def mutate_local(self, molecule):
        molecule, removed = removeAtoms(molecule, nmax=self.maxMutations)
        return molecule, removed

        #toss = np.random.rand(n_mutations)*self.cumProbs[-1]
        #i = np.searchsorted( self.cumProbs, toss )

        #print( "mutate_local ", i, toss, self.cumProbs[-1] )

        #molecule = moveMultipleAtoms(molecule, 1.0, 10)
        #for strat_nr in i:
        #    args = self.strategies[strat_nr][2]
        #    args['R'] = R
        #    molecule = self.strategies[strat_nr][1](molecule, p0, **args)
        #    #print(n_mutations, "mutations: ", self.strategies[strat_nr][1].__name__)

        #return molecule

        #args = self.strategies[i][2]
        #args['R'] = R
        #return self.strategies[i][1]( molecule, p0, **args )
        #return xyzs.copy(), Zs.copy(), qs.copy()