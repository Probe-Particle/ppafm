#!/usr/bin/python

import numpy as np
from . import elements
import math

# constants: 

bohrRadius2angstroem = 0.5291772109217 # find a good place for this

# additional procedures for CUBE files:

def loadNCUBE( fname ):
    f = open(fname )
    #First two lines of the header are comments
    header1=f.readline()
    header2=f.readline()
    #The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
    sth0 = f.readline().split()
    #The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
    sth1 = f.readline().split()
    sth2 = f.readline().split()
    sth3 = f.readline().split()
    f.close()
    return [ int(sth1[0]), int(sth2[0]), int(sth3[0]) ];

def loadCellCUBE( fname ):
    f = open(fname )
    #First two lines of the header are comments
    header1=f.readline()
    header2=f.readline()
    #The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
    line = f.readline().split()
    n0 = int(line[0])
    c0 =[ float(s) for s in line[1:4] ]

    #The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
    line = f.readline().split()
    n1 = int(line[0])
    c1 =[ float(s) for s in line[1:4] ]

    line = f.readline().split()
    n2 = int(line[0])
    c2 =[ float(s) for s in line[1:4] ]

    line = f.readline().split()
    n3 = int(line[0])
    c3 =[ float(s) for s in line[1:4] ]

#    cell0 = [c0[0]*   bohrRadius2angstroem, c0[1]   *bohrRadius2angstroem, c0[2]   *bohrRadius2angstroem]
    cell0 = [0.0, 0.0, 0.0]
    cell1 = [c1[0]*n1*bohrRadius2angstroem, c1[1]*n1*bohrRadius2angstroem, c1[2]*n1*bohrRadius2angstroem]
    cell2 = [c2[0]*n2*bohrRadius2angstroem, c2[1]*n2*bohrRadius2angstroem, c2[2]*n2*bohrRadius2angstroem]
    cell3 = [c3[0]*n3*bohrRadius2angstroem, c3[1]*n3*bohrRadius2angstroem, c3[2]*n3*bohrRadius2angstroem]
    f.close()
    return [ cell0, cell1, cell2, cell3 ];


# procedures for loading geometry from different files:

def loadAtoms( name ):
    f = open(name,"r")
    n=0;
    l = f.readline()
    #print "--",l,"--"
    try:
        n=int(l)
    except:
        raise ValueError("First line of a xyz file should contain the "
                         "number of atoms. Aborting...")
    if (n>0):
        n=int(l)
        e=[];x=[];y=[]; z=[]; q=[]
        i = 0;
        for line in f:
            words=line.split()
            nw = len( words)
            ie = None
            try:
                e.append( words[0] )
                x.append( float(words[1]) )
                y.append( float(words[2]) )
                z.append( float(words[3]) )
                if ( nw >=5 ):
                    q.append( float(words[4]) )
                else:
                    q.append( 0.0 )
            except:
                print(" skipped line : ", line)
    f.close()
    nDim = []
    lvec = [] 
    return [ e,x,y,z,q ], nDim, lvec

def loadXSFGeom( fname ):
    f = open(fname )
    e=[];x=[];y=[]; z=[]; q=[]
    nDim = []
    lvec = [] 
    for i in range(10000):
        if 'PRIMCOORD' in f.readline():
            break
    n = int(f.readline().split()[0])
    for j in range(n):
        ws = f.readline().split();  e.append(float(ws[0])); x.append(float(ws[1])); y.append(float(ws[2])); z.append(float(ws[3])); q.append(0);
    for i in range(10000):
        if 'BEGIN_DATAGRID_3D' in f.readline():    
            break
    ws = f.readline().split(); nDim = [int(ws[0]),int(ws[1]),int(ws[2])]
    for j in range(4):
        ws = f.readline().split(); lvec.append( [float(ws[0]),float(ws[1]),float(ws[2])] )
    f.close()
    print("nDim+1", nDim)
    nDim = np.array(nDim)-1
    #print "lvec", lvec
    #print "e,x,y,z", e,x,y,z
    return [ e,x,y,z,q ], nDim, lvec

def loadAtomsCUBE( fname ):
    e=[];x=[];y=[]; z=[]; q=[]
    f = open(fname )
    #First two lines of the header are comments
    header1=f.readline()
    header2=f.readline()
    #The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
    sth0 = f.readline().split()
    #The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector
    sth1 = f.readline().split()
    sth2 = f.readline().split()
    sth3 = f.readline().split()

    shift = [float(sth0[1]), float(sth0[2]), float(sth0[3])]
    nlines = int(sth0[0])
    for i in range(nlines):
        l=f.readline().split()
        r=[float(l[2]),float(l[3]),float(l[4])]
#        print l
        x.append( (r[0] - shift[0]) * bohrRadius2angstroem )
        y.append( (r[1] - shift[1]) * bohrRadius2angstroem )
        z.append( (r[2] - shift[2]) * bohrRadius2angstroem )
#        print float(l[2])*bohrRadius2angstroem, float(l[3])*bohrRadius2angstroem, float(l[4])*bohrRadius2angstroem
        e.append( int(  l[0]) )
        q.append(0.0)
    f.close()
    nDim = loadNCUBE( fname )
    lvec = loadCellCUBE( fname )
    return [ e,x,y,z,q ], nDim, lvec

def loadGeometryIN( fname ):
    print("importin atoms from FHI-AIMS input")
    f = open(fname )
    e=[];x=[];y=[]; z=[]; q=[]
    lvec = [] 
    for i in range(10000):
        ws = f.readline().split()
        if (len(ws)>0):
            if (ws[0]=='atom'):
                e.append(ws[4]); x.append(float(ws[1])); y.append(float(ws[2])); z.append(float(ws[3])); q.append(0);
            elif (ws[0]=='lattice_vector'):
                lvec.append([float(ws[1]),float(ws[2]),float(ws[3])])
            elif (ws[0]=='trust_radius'):
                break
    f.close()
    if (lvec != []):
        lvec = np.append([[0.,0.,0.]],lvec,  axis=0 )
    #print "lvec", lvec
    #print "e,x,y,z", e,x,y,z
    nDim = []
    return [ e,x,y,z,q ], nDim, lvec

# procedures for plotting:

def findBonds( atoms, iZs, sc, ELEMENTS = elements.ELEMENTS, FFparams=None ):
    bonds = []
    xs = atoms[1]
    ys = atoms[2]
    zs = atoms[3]
    n = len( xs )
    for i in range(n):
        for j in range(i):
            dx=xs[j]-xs[i]
            dy=ys[j]-ys[i]
            dz=zs[j]-zs[i]
            r=math.sqrt( dx*dx + dy*dy + dz*dz )
            ii = iZs[i]-1
            jj = iZs[j]-1    
            bondlength=ELEMENTS[FFparams[ii][2]-1][6]+ELEMENTS[FFparams[jj][2]-1][6]
            if (r<( sc * bondlength)) :
                bonds.append( (i,j) )
    return bonds

def findBondsSimple( xyz, rmax ):
    bonds = []
    xs = atoms[1]
    ys = atoms[2]
    zs = atoms[3]
    n = len( xs )
    for i in range(n):
        for j in range(i):
            dx=xs[j]-xs[i]
            dy=ys[j]-ys[i]
            dz=zs[j]-zs[i]
            r=math.sqrt(dx*dx+dy*dy+dz*dz)
            if (r<rmax) :
                bonds.append( (i,j) )
    return bonds

def getAtomColors( iZs, ELEMENTS = elements.ELEMENTS, FFparams=None ):
    colors=[]
    for e in iZs: 
        colors.append( ELEMENTS[ FFparams[e - 1][2] -1 ][7] )
    return colors

# other procedures for operating with geometries:

def multCell( xyz, cel, m=(2,2,1) ):
    n = len(xyz[0])
    mtot = m[0]*m[1]*m[2]*n
    es = [None] * mtot
    xs = [None] * mtot
    ys = [None] * mtot
    zs = [None] * mtot
    j  = 0
    for ia in range(m[0]):
        for ib in range(m[1]):
            for ic in range(m[2]):
                dx = ia*cel[0][0] + ib*cel[1][0] + ic*cel[2][0]
                dy = ia*cel[0][1] + ib*cel[1][1] + ic*cel[2][1]
                dz = ia*cel[0][2] + ib*cel[1][2] + ic*cel[2][2]
                for i in range(n):
                    es[j]=xyz[0][i]
                    xs[j]=xyz[1][i] + dx
                    ys[j]=xyz[2][i] + dy
                    zs[j]=xyz[3][i] + dz
                    j+=1
    return [es,xs,ys,zs]


# old procedure for loading geometry from bas: 

def loadBas(name):
    xyzs = []
    f = open(name,"r")
    while True:
        n=0;
        l = f.readline()
        #print "--",l,"--"
        try:
            n=int(l)
        except ValueError:
            break
        if (n>0):
            n=int(l)
            #f.readline()
            e=[];x=[];y=[]; z=[];
            for i in range(n):
                l=f.readline().split()
                #print l
                e.append( int(l[0]) )
                x.append( float(l[1]) )
                y.append( float(l[2]) )
                z.append( float(l[3]) )
            xyzs.append( [e,x,y,z] )
        else:
            break
        f.readline()
    f.close()
    return xyzs

def bas2xsf(iZs,Rs,nPBC):
    div=(nPBC[0]*2+1)*(nPBC[1]*2+1) # amount of single unit cells
    li=int(len(iZs)/div)            # amount of atoms in single cell
    sa=li*(nPBC[0]*2+1)+li*nPBC[1]        # starting atom to plot
    ea=sa+li                         # ending atom to plot
    header1='''ATOMS \n'''
    header2=''
    for ia in range(sa,ea):
        header2+=str(iZs[ia])+' '+str(Rs[ia][0])+' '+str(Rs[ia][1])+' '+str(Rs[ia][2])+' \n'
    header3='''
BEGIN_BLOCK_DATAGRID_3D                        
   some_datagrid      
   BEGIN_DATAGRID_3D_whatever 
'''
    #print ("header1+header2+header3")
    #print (header1+header2+header3)
    #print ("headers written")
    return header1+header2+header3 ;
