#!/usr/bin/python

#from . import elements
import elements
import math
import numpy as np

#from sortedcontainers import SortedDict

#exclude_default = set(1)

# ===========================
#      Molecular Topology
# ===========================

iDebug=0

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
    ngs = [ [] for i in range(na) ]
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
            for k in sorted(dct.keys()):
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

'''
def findTrisRings(bonds,neighs):
    tris = {}
    for ia,ib in bonds:
        a_ngs  = neighs[ia]
        b_ngs  = neighs[ib]
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
'''

def findTris(bonds,neighs):
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
            if(iDebug>0): print("WARRNING: bond ", b, " common neighbors ", common)
            continue
        elif ncm<1:
            if(iDebug>0): print("WARRNING: bond ", b, " common neighbors ", common)
            continue
        tri0 = tuple(sorted(b+(common[0],)))
        tris.add(tri0)
        if len(common)==2:
            tri1 = tuple(sorted(b+(common[1],)))
            tris.add(tri1)
            tbonds.append((tri0,tri1))
    return tris, tbonds


def findTris_(bonds,neighs):
    tris   = set()
    tbonds = []
    #bset = set(bonds)
    #bset = { tuple(sorted(b)) for b in bonds }
    #print "bset ", bset
    #exit()
    for b in bonds:
    #for b in bset:
        a_ngs  = neighs[b[0]]
        b_ngs  = neighs[b[1]]
        common = []
        for i in a_ngs:
            if i in b_ngs:
                common.append(i)
        #print "bond ",b," common ",common
        ncm = len(common)
        if   ncm>2:
            #print "WARRNING: bond ", b, " common neighbors ", common
            continue
        elif ncm<1:
            #print "WARRNING: bond ", b, " common neighbors ", common
            continue
        '''
        elif (len(common)==2):
            cmn  = tuple(sorted(common)) 
            #print cmn
            if cmn in bset :
                print "WARRNING: bond ", b, " corss bond ", cmn
                if cmn[0] > b[0]:
                    print "removed "
                    continue
        '''
        tri0 = tuple(sorted(b+(common[0],)))
        tris.add(tri0)
        if len(common)==2:
            tri1 = tuple(sorted(b+(common[1],)))
            tris.add(tri1)
            tbonds.append((tri0,tri1))
    return tris, tbonds

def getRingNatom(atom2ring,nr):
    #nr = len(ringNeighs)
    nra=np.zeros(nr,dtype=np.int)
    for r1,r2,r3 in atom2ring:
        nra[r1]+=1
        nra[r2]+=1
        nra[r3]+=1
    return nra

'''
def selectNringAtoms(atom2ring,ringNeighs,N=6):
    na=len(atom2ring)
    mask=np.empty(na,dtype=np.bool)
    for ia in xrange(na):
        a1,a2,a3=atom2ring[ia]
        n1=len(ringNeighs[a1])
        n2=len(ringNeighs[a2])
        n3=len(ringNeighs[a3])
        #print n1,n2,n3
        mask[ia]=((n1==N)or(n2==N)or(n3==N))
    #print "N6mask ", mask ; exit()
    return mask
'''

def tris2num_(tris, tbonds):
    t2i     = { k:i for i,k in enumerate(tris) }
    tbonds_ = [ (t2i[i],t2i[j]) for i,j in tbonds ]
    return tbonds_,t2i

def trisToPoints(tris,ps):
    ops=np.empty((len(tris),2))
    for i,t in enumerate(tris):
        ops[i,:] = (ps[t[0],:] + ps[t[1],:] + ps[t[2],:])/3.0
        #ops.append()
    return ops

def removeBorderAtoms(ps,cog,R):
    rnds = np.random.rand(len(ps))
    r2s  = np.sum((ps-cog[None,:])**2, axis=1)  #;print "r2s ", r2s, R*R 
    mask = rnds > r2s/(R*R)
    return mask

def validBonds( bonds, mask, na ):
    a2a = np.cumsum(mask)-1
    bonds_ = []
    #print mask
    for i,j in bonds:
        #print i,j
        if( mask[i] and mask[j] ):
            bonds_.append( (a2a[i],a2a[j]) )
    return bonds_

'''
def tris2num(tris):
    t2i = { k:i for i,k in enumerate(tris) }
    out = []
    for k,v in tris:
        out.append( [ t2i[t] for t in v] )
    return out
'''


# ===========================
#
# ==========================

def removeAtoms( atom_pos, bonds, atom2ring, cog, Lrange=10 ):
    # --- remove some atoms
    mask      = removeBorderAtoms(atom_pos,cog,Lrange)          # ;print "remove atom mask ", mask
    bonds     = validBonds( bonds, mask, len(atom_pos) )   # ;print "rm:bonds ", bonds
    atom_pos  = atom_pos [mask,:]                             # ;print "rm:atom_pos ", atom_pos
    atom2ring = atom2ring[mask,:]                             # ;print "rm:atom2ring ", atom2ring
    return atom_pos, bonds, atom2ring

def ringsToMolecule( ring_pos, ring_Rs, Lrange=6.0 ):
    Nring = len( ring_pos )
    cog = np.sum(ring_pos,axis=0)/Nring
    ring_bonds  = findBonds(ring_pos,ring_Rs,fR=1.0)
    ring_neighs = bonds2neighs(ring_bonds,Nring)
    ring_nngs   = np.array([ len(ng) for ng in ring_neighs ],dtype=np.int) # print "ring_nngs ", ring_nngs

    # --- atoms are centers of triangles with vertexes in ring center
    tris,bonds_ = findTris(ring_bonds,ring_neighs)
    atom2ring   = np.array( list(tris), dtype=np.int )     #; print "atom2ring ", atom2ring

    atom_pos = ( ring_pos[atom2ring[:,0]] + ring_pos[atom2ring[:,1]] + ring_pos[atom2ring[:,2]] )/3.0
    bonds,_  = tris2num_(tris, bonds_)
    
    atom_pos, bonds, atom2ring, = removeAtoms( atom_pos, bonds, atom2ring, cog, Lrange )
    bonds = np.array(bonds)
    
    # --- select aromatic hexagons as they have more pi-character 
    ring_natm   = getRingNatom(atom2ring,len(ring_neighs))          # ;print "ring_natm ", ring_natm
    ring_N6mask = np.logical_and( ring_natm[:]==6, ring_nngs[:]==6 )   # ;print "ring_N6mask", len(ring_N6mask), ring_N6mask 
    atom_N6mask = np.logical_or( ring_N6mask[atom2ring[:,0]], 
                  np.logical_or( ring_N6mask[atom2ring[:,1]], 
                                 ring_N6mask[atom2ring[:,2]]  ) )      # ;print "atom_N6mask", len(atom_N6mask), atom_N6mask

    neighs  = bonds2neighs( bonds, len(atom_pos) )    # ;print neighs
    nngs    = np.array([ len(ngs) for ngs in neighs ],dtype=np.int) 
    
    atypes=nngs.copy()-1
    atypes[atom_N6mask]=3       #; print "atypes ", atypes
    
    return atom_pos,bonds, atypes, nngs, neighs, ring_bonds, atom_N6mask





# ===========================
#   Atom Types and groups
# ===========================

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
    #print "levels", levels
    for i,nng in enumerate(nngs):
        #print i,nng
        ing = nng-1
        il = np.searchsorted( levels[ing], rnds[i]  )
        #print i, nng, il, rnds[i], levels[ing]  #, levels[nng][il]
        elist.append(species[ing][il][0])
    return elist

def makeGroupLevels(groupDict):
    for k,groups in groupDict.items():
        vsum=0
        #print "k,groups ", k,groups
        l = np.empty(len(groups))
        for i,(g,v) in enumerate(groups):
            vsum+=v
            l[i] =vsum
        l/=vsum
        #print l
        groupDict[k] = [l,] + groups
    return groupDict

def selectRandomGroups( an, ao, groupDict ):
    na = len(an)
    rnds=np.random.rand(na)
    out = []
    #print "levels", levels
    atoms = []
    for i in range(na):
        k = (an[i],ao[i])
        if k in groupDict:
            groups = groupDict[k]
            levels = groups[0]
            #print levels
            il     = np.searchsorted( levels, rnds[i] )
            out.append( groups[il+1][0] )
        else:
            out.append( None )
        print(k, out[-1])
    return out


group_definition = {
# name   Center,  ndir,  nb,nsigma,npi  nt,nH,ne
#         0   1  2 3 4  5 6 7
"-CH3" :("C" ,4, 1,1,0, 3,3,0),
"-NH2" :("N" ,4, 1,1,0, 3,2,1),
"-OH"  :("O" ,4, 1,1,0, 3,1,2),
"-F"   :("F" ,4, 1,1,0, 3,0,3),
"-Cl"  :("Cl",4, 1,1,0, 3,0,3),

"-CH2-":("C", 4, 2,2,0, 2,2,0),
"-NH-" :("N", 4, 2,2,0, 2,1,1),
"-O-"  :("O", 4, 2,2,0, 2,0,2),

"*CH"  :("C", 4, 3,3,0, 1,1,0),
"*N"   :("N", 4, 3,3,0, 1,0,1),   # what about N+ ?

"=CH-" :("C", 3, 3,2,1, 1,1,0),
"=N-"  :("N", 3, 3,2,1, 1,0,1),

"=CH2" :("C" ,3, 2,1,1, 2,2,0),
"=NH"  :("N" ,3, 2,1,1, 2,1,1),
"=O"   :("O" ,3, 2,1,1, 2,0,2),

"*C"   :("C", 3, 4,3,1, 0,0,0),
"*N+"  :("N", 3, 4,3,1, 0,0,0), 

"#CH"  :("C", 2, 3,1,2, 1,1,0),
"#N"   :("N", 2, 3,1,2, 1,0,1),
}

'''
Simplified

nt = nH
nb = nsigma + npi

4 = nb + nt
4 = nsigma + npi + nH + ne

ndir = nsigma + nH + ne
ndir = nsigma + nt

'''

def normalize(v):
    l  = np.sqrt(np.dot(v,v))
    v/=l 
    return v,l

def makeTetrahedron(db,up):
    #phi = np.random()
    normalize(db)
    side = np.cross(db,up)
    # https://en.wikipedia.org/wiki/Tetrahedron#Formulas_for_a_regular_tetrahedron
    a = 0.81649658092  # sqrt(2/3)
    b = 0.47140452079  # sqrt(2/9)
    c = 0.33333333333  # 1/3
    return np.array([ db*c + up*(b*2)     ,
                      db*c - up*b - side*a,
                      db*c - up*b + side*a, ])

def makeTetrahedronFork(d1,d2):
    up   = np.cross(d1,d2);  normalize(up)
    db   = d1+d2;            normalize(db)
    #side = np.cross(d,up)
    #a = 0.5           # 1/2
    #b = 0.35355339059 # sqrt(1/8)
    a = 0.81649658092  # sqrt(2/3)
    b = 0.57735026919  # sqrt(1/3)
    return np.array([ db*b + up*a,
                      db*b - up*a, ])

def makeTriFork(db,up):
    normalize(db)
    side = np.cross(db,up)
    a = 0.87758256189  # 1/2
    b = 0.5            # sqrt(1/8)
    return np.array([ db*b + side*a,
                      db*b - side*a, ])

def groups2atoms( groupNames, neighs, ps ):

    def appendHs( txyz, Hmask, elems, xyzs, e1="H", e2="He" ):
        Hm = Hmask[ np.random.randint(len(Hmask)) ]
        #print Hm
        #if len(Hm) == 1:
        #    print Hm, txyz 
        for ih in range(len(Hm)):
            if Hm[ih] == 1:
                elems.append( e1     )
                xyzs .append( txyz[ih] )
            else:
                elems.append( e2     )
                xyzs .append( txyz[ih] )

    up=np.array((0.,0.,1.))
    Hmasks3=[ [(0,0,0)],
              [(1,0,0),(0,1,0),(0,0,1)],
              [(0,1,1),(1,0,1),(1,1,0)],
              [(1,1,1)]]
    Hmasks2=[[(0,0)],
             [(1,0),(1,0)],
             [(1,1)]]
    elems=[]; xyzs=[]
    for ia,name in enumerate(groupNames):
        if name in group_definition:
            ngs = neighs[ia]
            pi  = ps[ia]
            g   = group_definition[name]
            ndir   = g[1]
            nsigma = g[3]
            nH     = g[6]
            elems.append(g[0])
            xyzs.append(pi.copy())
            
            #print name, ndir, nsigma, nH
            
            if      ( ndir==4 ):  # ==== tetrahedral
                flip = np.random.randint(2)*2-1
                #print "---------- flip ", flip
                if   (nsigma==1):     # like -CH3
                    print(name, ndir, nsigma, nH)
                    txyz = makeTetrahedron( pi-ps[ngs[0]] , up*flip ) + pi[None,:]
                    appendHs( txyz, Hmasks3[nH], elems, xyzs )
                elif (nsigma==2):     # like -CH2-
                    txyz = makeTetrahedronFork( pi-ps[ngs[0]], pi-ps[ngs[1]] ) + pi[None,:]
                    appendHs( txyz, Hmasks2[nH], elems, xyzs )
                elif (nsigma==3):     # like *CH
                    if nH==1:
                        elems.append("H")
                        xyzs.append(pi+up*flip)
                    
            elif ( ndir==3 ):  # ==== triangular
                if   (nsigma==1):     # like =CH2
                    txyz  = makeTriFork(pi-ps[ngs[0]],up) + pi[None,:]
                    appendHs( txyz, Hmasks2[nH], elems, xyzs )
                elif (nsigma==2):    # like  =CH-
                    appendHs(  normalize( pi*2 - ps[ngs[0]] - ps[ngs[1]] )[0] + pi[None,:] , [(1,)], elems, xyzs )
                    #if nH==1:
                    #    elems.append("H")
                    #    xyzs.append( pi + ( normalize( pi*2 - ps[ngs[0]] - ps[ngs[1]] )[0] ) )
                    
            elif ( ndir==2 ):  # ==== linear
                appendHs( normalize( pi-ps[ngs[0]] )[0] + pi[None,:], [(1,) ], elems, xyzs ) 
                #if   (nsigma==1):
                #    elems.append("H")
                #    xyzs.append( pi + ( normalize( pi-ps[ngs[0]] )[0] ) )
            
        else:
            print("Group >>%s<< not known" %name)
    print("len(xyzs), len(elems) ", len(xyzs), len(elems))
    for xyz in xyzs: 
        print(len(xyz), end=' ')
        if len(xyz) != 3:
            print(xyz)
    return np.array(xyzs), elems

# ===========================
#           FIRE
# ===========================

class FIRE:
    v = None
    minLastNeg   = 5
    t_inc        = 1.1
    t_dec        = 0.5
    falpha       = 0.98
    kickStart    = 1.0
    
    def __init__(self, dt_max=0.2, dt_min=0.01, damp_max=0.2, f_limit=10.0, v_limit=10.0 ):
        self.dt       = dt_max
        self.dt_max   = dt_max
        self.dt_min   = dt_min
        self.damp     = damp_max
        self.damp_max = damp_max
        self.v_limit  = v_limit
        self.f_limit  = f_limit
        self.bFIRE    = True
        #self.bFIRE    = False
        
        self.lastNeg = 0
    
    def move(self,p,f):
        if self.v is None:
            self.v=np.zeros(len(p))
        v=self.v
        
        f_norm = np.sqrt( np.dot(f,f) )
        v_norm = np.sqrt( np.dot(v,v) )
        vf     =          np.dot(v,f)
        dt_sc  = min( min(1.0,self.f_limit/(f_norm+1e-32)), min(1.0,self.v_limit/(v_norm+1e-32)) )
        
        if self.bFIRE:
            if ( vf < 0.0 ) or ( dt_sc < 0.0 ):
                self.dt      = max( self.dt * self.t_dec, self.dt_min );
                self.damp    = self.damp_max
                self.lastNeg = 0
                v[:] = f[:]* self.dt * dt_sc
            else:
                v[:] = v[:]*(1-self.damp) +  f[:]*( self.damp * v_norm/(f_norm+1e-32) )
                if self.lastNeg > self.minLastNeg:
                    self.dt   = min( self.dt * self.t_inc, self.dt_max );
                    self.damp = self.damp  * self.falpha;
                self.lastNeg+=1
        else:
            v[:] *= (1-self.damp_max)
        
        dt_ = self.dt * dt_sc
        v[:] += f[:]*dt_
        p[:] += v[:]*dt_
        #print "|f|,dt,damp,cvf,dt_sc", f_norm, self.dt, self.damp, vf/(v_norm*f_norm), dt_sc
        return f_norm

# ===========================
#           Bond-Order Opt
# ===========================

def simpleAOEnergies(  Eh2=-4,Eh3=4,   E12=0,E13=0,   E22=0,E23=1,   E32=0,E33=4,  Ex1=0., Ex4=10., Ebound=20. ):
    typeEs = np.array([
     [Ebound,Ex1,E12,E13,Ex4,Ebound], # nng=1
     [Ebound,Ex1,E22,E23,Ex4,Ebound], # nng=2
     [Ebound,Ex1,E32,E33,Ex4,Ebound], # nng=3
     [Ebound,Ex1,Eh2,Eh3,Ex4,Ebound], # hex
    ])
    return typeEs

def assignAtomBOFF(atypes, typeEs):
    from scipy.interpolate import Akima1DInterpolator
    nt=len(typeEs)
    na=len(atypes)
    typeMasks = np.empty((nt,na),dtype=np.bool)
    #typeSelects = []
    typeFFs = []
    Xs = np.array([-1,0,1,2,3,4])
    #print "typeEs ",typeEs
    #print "atypes ",atypes
    for it in range(nt):
        typeMasks[it,:] = ( atypes[:] == it )
        Efunc = Akima1DInterpolator(Xs,typeEs[it])
        Ffunc = Efunc.derivative()
        typeFFs.append(Ffunc) 
        #print  "mask[%i]" %it, masks[it,:]
    return typeMasks, typeFFs 

def relaxBondOrder( bonds, typeMasks, typeFFs, fConv=0.01, nMaxStep=1000, EboStart=0.0, EboEnd=10.0, boStart=None, optimizer=None ):
    # print " ==== ", EboStart, EboEnd
    #ao=np.zeros(len(nngs ),dtype=np.int)
    #nngs=np.array(nngs)
    nt=typeMasks.shape[0]
    na=typeMasks.shape[1]
    nb=len(bonds)
    if boStart is None:
        bo=np.zeros(nb) + 0.33 # initial guess
    else:
        bo=boStart.copy()
    fb=np.empty(nb)
    #vb=np.zeros(nb)
    fa=np.empty(na)
    ao=np.empty(na)        # + 0.5 # initial guess
    #exit(0)
    
    if optimizer is None:
        optimizer = FIRE()
    
    #print "bo0 ", bo[:6]
    
    #f_debug = []
    for itr in range(nMaxStep):
        # -- Eval Atom derivs
        #fa[:] = 0
        #fb[:] = 0
        
        # -- update Atoms
        ao[:] = 0
        for ib,(i,j) in enumerate(bonds):
            boi = bo[ib]
            ao[i] += boi
            ao[j] += boi
        
        for it in range(nt):
            #typeFs[mask]
            Ffunc     = typeFFs[it]
            mask      = typeMasks [it] 
            #sel      = typeSelects[it]
            fa[mask]  = Ffunc( ao[mask] )
        # -- Eval Bond derivs
        #for ib,(i,j) in enumerate(bonds):
        #    fb[ib] = fa[i] + fa[j]
        fb  = fa[bonds[:,0]] + fa[bonds[:,1]]
        Ebo = (EboEnd-EboStart)*(itr/float(nMaxStep-1)) + EboStart
        fb += Ebo*np.sin(bo*np.pi*2)   # force integer forces
        # -- move
        #bo -= fb*dt
        #vb[:]  = vb[:]*(1-damping) - fb[:]*dt
        #bo[:] += vb[:]*dt
        
        #print "bo[:6]", bo[:6]
        #print "fb[:6]", fb[:6]
        #print itr,Ebo,
        fb[:]*=-1
        f_norm = optimizer.move(bo,fb)
        
        if f_norm < fConv:
            break
        
        #f_debug.append(f_norm)
        
        #print itr,f_norm
        #print " bo[:6]", bo[:6]
        #print " fb[:6]", fb[:6]
        
        #ao[bonds[:,0]] += bo
        #ao[bonds[:,1]] += bo
        
        #print "bo ", bo,"\n fb ", fb
        '''
        nview=6
        print "---- itr --- ", itr
        print "ao ", ao[:nview]
        print "bo ", bo[:nview]
        print "vb ", vb[:nview]
        print "fb ", fb[:nview]
        '''
    
    #print "f_debug", f_debug
    #import matplotlib.pyplot as plt 
    #plt.figure()
    #plt.plot(f_debug); plt.yscale('log')
    
    #print "boEnd ", bo[:6]
    return bo,ao

def estimateBondOrder( atypes, bonds, E12=0.5, E22=+0.5, E32=+0.5 ):
    typeEs = simpleAOEnergies( E12=E12, E22=E22, E32=E32 )
    typeMasks, typeFFs = assignAtomBOFF(atypes, typeEs)
    opt   = FIRE(dt_max=0.1,damp_max=0.25)
    bo,ao = relaxBondOrder( bonds, typeMasks, typeFFs, fConv= 0.0001          , optimizer=opt, EboStart=0.0,  EboEnd=0.0                ) # relax delocalized pi-bond superposition
    bo,ao = relaxBondOrder( bonds, typeMasks, typeFFs, fConv=-1., nMaxStep=100, optimizer=opt, EboStart=0.0,  EboEnd=10.0 , boStart=bo  ) # gradually increase integer-discretization strenght 'Ebo'
    bo,ao = relaxBondOrder( bonds, typeMasks, typeFFs, fConv=0.0001           , optimizer=opt, EboStart=10.0, EboEnd=10.0 , boStart=bo  ) # relax finaly with maximum discretization strenght 'Ebo'
    return bo,ao, typeEs

# ===========================
#       Geometry Opt
# ===========================

def getForceIvnR24( ps, Rs ):
    r2safe = 1e-4
    #if fs is None:
    #    fs = np.zeros(ps.shape)
    #if ds is None:
    #    ds = np.zeros(ps.shape)
    na    = len(ps)
    ds    = np.zeros(ps.shape)
    fs    = np.zeros(ps.shape)
    ir2s  = np.zeros(na    )
    R2ijs = np.zeros(na    )
    for i in range(na):
        R2ijs      = Rs[:]+Rs[i]
        R2ijs[:]  *= R2ijs[:]
        ds[:,:]    = ps - ps[i][None,:]
        ir2s[:]    = 1/(np.sum(ds**2,axis=1) + r2safe)
        ir2s[i]    = 0
        fs  [:,:] += ds[:,:]*((R2ijs*ir2s-1)*R2ijs*ir2s*ir2s)[:,None]
    return fs
#def tris2skelet(tris,):

def relaxAtoms( ps, aParams, FFfunc=getForceIvnR24, fConv=0.001, nMaxStep=1000, optimizer=None ):
    if optimizer is None:
        optimizer = FIRE()
    
    f_debug = []
    for itr in range(nMaxStep):
        fs = FFfunc(ps,aParams)
        f_norm = optimizer.move(ps.flat,fs.flat)
        #print fs[:6]
        #print itr, f_norm
        if f_norm < fConv:
            break
        f_debug.append(f_norm)
        
    #import matplotlib.pyplot as plt 
    #plt.figure()
    #plt.plot(f_debug); plt.yscale('log')
    
    return ps

