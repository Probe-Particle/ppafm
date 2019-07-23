#!/usr/bin/python

import elements
import math
import numpy as np

#from sortedcontainers import SortedDict

#exclude_default = set(1)

def findBonds( xyzs, Zs, ELEMENTS=elements.ELEMENTS, Rcut=2.0, fRvdw=1.3 ):
    #n     = len(Zs)
    n     = len(xyzs)
    #print "len(xyzs)", n
    bonds = []
    R2cut = Rcut*Rcut
    inds  = np.indices((n,))[0]
    #print inds
    Rvdws = np.array([ ELEMENTS[iz-1][6]*fRvdw for iz in Zs ])
    #print "RvdWs: ",Rvdws
    for i in range(1,n):
        ds     = xyzs[:i,:] - xyzs[i,:][None,:]
        #print "ds.shape ", ds.shape
        r2s   = np.sum( ds**2, axis=1 ); #print r2s
        #rs    = np.sqrt( np.sum( ds**2, axis=1 ) ); print rs
        mask   = r2s < ( Rvdws[:i] + Rvdws[i] )**2
        #mask   = (rs - Rvdws[:i]) <  + Rvdws[i] 
        #print "mask.shape ", mask.shape
        #print "inds.shape ", inds.shape
        sel    = inds[:i][mask]
        bonds += [ (i,j) for j in sel ]
    return bonds

def bonds2neighs( bonds, Zs ):
    ngs = [ [] for i in Zs ]
    for i,j in bonds:
        ngs[i].append((j,Zs[j]))
        ngs[j].append((i,Zs[i]))
    return ngs

def neighs2str( Zs, neighs, ELEMENTS=elements.ELEMENTS, bPreText=False ):
    groups = [ '' for i in Zs ]
    for i,ngs in enumerate(neighs):
        nng = len(ngs)
        if nng > 1:
            #s = ELEMENTS[Zs[i]-1][1] + ("(%i):" %nng)
            if bPreText:
                s = ELEMENTS[Zs[i]-1][1]+":"
            else:
                s = ""
            dct = {}
            for j,jz in ngs:
                if jz in dct:
                    dct[jz] += 1
                else:
                    dct[jz] =1
            for k in sorted(dct.iterkeys()):
                s+= ELEMENTS[k-1][1] + str(dct[k])
            groups[i] = s
    return groups

'''
def classifyGroups( Zs, neighs ):
    groups = [ '' for i in Zs ]
    for i,iz in enumerate(Zs):
        nng = len(neighs[i])
        if   iz==6: # carbon
            if nng == 4:
            if nng == 3:
        elif iz==7: # nitrogen
            
        elif iz==8: # oxygen
'''
