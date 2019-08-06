#!/usr/bin/python

import elements
import math
import numpy as np

#from sortedcontainers import SortedDict

#exclude_default = set(1)

def findBonds( xyzs, Rs, fR=1.3 ):
    n     = len(xyzs)
    bonds = []
    inds  = np.indices((n,))[0]
    for i in range(1,n):
        ds     = xyzs[:i,:] - xyzs[i,:][None,:]
        r2s    = np.sum( ds**2, axis=1 )
        mask   = r2s < ( (Rs[:i] + Rs[i])*fR )**2
        sel    = inds[:i][mask]
        bonds += [ (i,j) for j in sel ]
    return bonds

def findBondsZs( xyzs, Zs, ELEMENTS=elements.ELEMENTS, fR=1.3 ):
    Rs = np.array([ ELEMENTS[iz-1][6]*fRvdw for iz in Zs ])
    findBonds( xyzs, Rs, fR=1.3 )
    return findBonds( xyzs, Rs, fR=fR )

'''
def findBondsZs( xyzs, Zs, ELEMENTS=elements.ELEMENTS, Rcut=2.0, fRvdw=1.3 ):
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
'''

def bonds2neighs( bonds, na ):
    ngs = [ [] for i in xrange(na) ]
    for i,j in bonds:
        ngs[i].append(j)
        ngs[j].append(i)
    return ngs

def bonds2neighsZs( bonds, Zs ):
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


'''
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
'''

#def orderTriple(a,b,c):
#    if a>b:
#        

'''
def tryInsertTri_i(tris,tri,i):
    if tri in tris:
        tris[tri].append(i)
    else:
        tris[tri] = [i]

def tryInsertTri(tris,tri):
    #print "tri ", tri
    if not (tri in tris):
        tris[tri] = []
'''

def findTris(bonds,neighs):
    tris = {};
    for b in bonds:
        a_ngs  = neighs[b[0]]
        b_ngs  = neighs[b[1]]
        common = []
        for i in a_ngs:
            if i in b_ngs:
                common.append(i)
        #print "bond ",b," common ",common
        ncm = len(common)
        if   ncm>2:
            print "WARRNING: bond ", b, " common neighbors ", common
            continue
        elif ncm<1:
            print "WARRNING: bond ", b, " common neighbors ", common
            continue
        tri0 = tuple(sorted(b+(common[0],)))
        if len(common)==2:
            tri1 = tuple(sorted(b+(common[1],)))
            tris.setdefault(tri1,[]).append( tri0 )
            tris.setdefault(tri0,[]).append( tri1 )
            #print tri0, tri1
            #tris.setdefault(tri0,[]).append(common[1])
            #tris.setdefault(tri1,[]).append(common[0])
            #setdefault()
            #tryInsertTri_i(tris,tri0,common[1])
            #tryInsertTri_i(tris,tri1,common[0])
        else:
            tris.setdefault(tri0,[])
            #tryInsertTri(tris,tri0)
    return tris

def findTris_(bonds,neighs):
    tris   = set()
    tbonds = []
    for b in bonds:
        a_ngs  = neighs[b[0]]
        b_ngs  = neighs[b[1]]
        common = []
        for i in a_ngs:
            if i in b_ngs:
                common.append(i)
        #print "bond ",b," common ",common
        ncm = len(common)
        if   ncm>2:
            print "WARRNING: bond ", b, " common neighbors ", common
            continue
        elif ncm<1:
            print "WARRNING: bond ", b, " common neighbors ", common
            continue
        tri0 = tuple(sorted(b+(common[0],)))
        tris.add(tri0)
        if len(common)==2:
            tri1 = tuple(sorted(b+(common[1],)))
            tris.add(tri1)
            tbonds.append((tri0,tri1))
    return tris, tbonds

def trisToPoints(tris,ps):
    ops=np.empty((len(tris),2))
    for i,t in enumerate(tris):
        ops[i,:] = (ps[t[0],:] + ps[t[1],:] + ps[t[2],:])/3.0
        #ops.append()
    return ops

def tris2num_(tris, tbonds):
    t2i     = { k:i for i,k in enumerate(tris) }
    tbonds_ = [ (t2i[i],t2i[j]) for i,j in tbonds ]
    return tbonds_,t2i

'''
def tris2num(tris):
    t2i = { k:i for i,k in enumerate(tris) }
    out = []
    for k,v in tris:
        out.append( [ t2i[t] for t in v] )
    return out
'''

def normalizeSpeciesProbs( species ):
    out = []
    
    for l in species:
        #psum = 0
        #for s in l:
        #    psum+=s[1]
        renorm=1.0/sum( s[1] for s in l )
        #out.append([ (s[0],s[1]*renorm) for s in l ])
    return out

def speciesToPLevels( species ):
    levels = []
    for l in species:
        l_ = [s[1]*1.0 for s in l]
        l_ = np.cumsum(l_)
        l_*=(1.0/l_[-1])
        levels.append(l_)
        #psum = 0
        #for s in l:
        #    psum+=s[1]
        #renorm=1.0/sum( s[1] for s in l )
        #levels.append( np.cumsum(l)*renorm )
        #out.append([ (s[0],s[1]*renorm) for s in l ])
    return levels

def selectRandomElements( nngs, species, levels ):
    rnds=np.random.rand(len(nngs))
    elist = []
    print "levels", levels
    for i,nng in enumerate(nngs):
        #print i,nng
        ing = nng-1
        il = np.searchsorted( levels[ing], rnds[i]  )
        print i, nng, il, rnds[i], levels[ing]  #, levels[nng][il]
        elist.append(species[ing][il][0])
    return elist


'''
def bondOrders( nngs, bonds, Nstep=100 ):
    #ao=np.zeros(len(nngs ),dtype=np.int)
    ae=np.zeros(len(nngs ))
    ao=nngs.copy()
    bo=np.zeros(len(bonds),dtype=np.int)
    for itr in xrange(Nstep):
'''



def bondOSmooth( nngs, bonds, Nstep=100, EboMax=100 ):
    #ao=np.zeros(len(nngs ),dtype=np.int)
    ae=np.zeros(len(nngs ))
    ao=nngs.copy()
    bo=np.zeros(len(bonds),dtype=np.int)
    for itr in xrange(Nstep):
        E = 0


#def tris2skelet(tris,):

