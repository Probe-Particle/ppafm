#!/usr/bin/python

import os
import re
import math
import numpy as np

from . import elements

verbose = 0

def loadXYZ(fname):
    '''
    Read the contents of an xyz file.
    
    The standard xyz file format only has per-atom elements and xyz positions. In Probe-Particle
    we also use the per-atom charges, which can be written as an extra column into the xyz file.
    By default the fifth column is interpreted as the charges, but if the file is written in the 
    extended xyz format used by ASE, the relevant column indicated in the comment line is used.

    Arguments:
        fname: str. Path to file.

    Returns:
        xyzs: np.ndarray of shape (N_atoms, 3). Atom xyz positions.
        Zs: np.ndarray of shape (N_atoms,). Atomic numbers.
        qs: np.ndarray of shape (N_atoms). Per-atom charges. All zeros if no charges are present in the file.
        comment: str. The contents of the second line of the xyz file.
    '''

    xyzs = [] 
    Zs = []
    extra_cols = []

    with open(fname, 'r') as f:

        line = f.readline().strip()
        try:
            N = int(line)
        except ValueError:
            raise ValueError(f'The first line of an xyz file should have the number of atoms, but got `{line}`')
        
        comment = f.readline().strip()
        
        for i, line in enumerate(f):
            if i >= N: break
            wds = line.split()
            try:
                Z = wds[0]
                if Z in elements.ELEMENT_DICT:
                    Z = elements.ELEMENT_DICT[Z][0]
                else:
                    Z = int(Z)
                xyzs.append( ( float(wds[1]), float(wds[2]), float(wds[3]) ) )
                Zs.append(Z)
                extra_cols.append(wds[4:])
            except (ValueError, IndexError):
                raise ValueError(f'Could not interpret line in xyz file: `{line}`')

    xyzs = np.array(xyzs, dtype=np.float64)
    Zs = np.array(Zs, dtype=np.int32)
        
    if len(extra_cols[0]) > 0:
        qs = _getCharges(comment, extra_cols)
    else:
        qs = np.zeros(len(Zs), dtype=np.float64)

    return xyzs, Zs, qs, comment

def _getCharges(comment, extra_cols):
    match = re.match('.*Properties=(\S*) ', comment)
    if match:
        # ASE format, check if one of the columns has charges
        props = match.group(1).split(':')[6:] # [6:] is for skipping over elements and positions
        col = 0
        for name, size in zip(props[::3], props[2::3]):
            if name in ['charge', 'initial_charges']:
                qs = np.array([float(ex[col]) for ex in extra_cols], dtype=np.float64)
                break
            col += int(size)
        else:
            qs = np.zeros(len(extra_cols), dtype=np.float64)
    else:
        # Not ASE format, so just take first column
        qs = np.array([float(ex[0]) for ex in extra_cols], dtype=np.float64)
        if _notActuallyCharge(qs):
            # qs is not actually charges based on some heuristics
            qs = np.zeros(len(qs), dtype=np.float64)
    return qs

def _notActuallyCharge(qs):
    if abs(sum(qs)) > 3:
        return True
    if np.abs(qs).max() > 3:
        return True
    return False

def saveXYZ(fname, xyzs, Zs, qs=None, comment='', append=False):
    '''
    Save atom types, positions, and, (optionally) charges to an xyz file.

    Arguments:
        fname: str. Path to file.
        xyzs: np.ndarray of shape (N_atoms, 3). Atom xyz positions.
        Zs: np.ndarray of shape (N_atoms,). Atom atomic numbers.
        qs: np.ndarray of shape (N_atoms) or None. If not None, the partial charges of atoms written as
            the fifth column into the xyz file.
        comment: str. Comment string written to the second line of the xyz file.
        append: bool. Append to file instead of overwriting if it already exists. Useful for creating
            movies of changing structures.
    '''
    N = len(xyzs)
    mode = 'a' if append else 'w'
    file_exists = os.path.exists(fname)
    with open(fname, mode) as f:
        if append and file_exists: f.write('\n')
        f.write(f'{N}\n{comment}\n')
        for i in range(N):
            f.write(f'{Zs[i]} {xyzs[i, 0]} {xyzs[i, 1]} {xyzs[i, 2]}')
            if qs is not None: f.write(f' {qs[i]}')
            if i < (N-1): f.write('\n')

def loadGeometryIN(fname):
    Zs = []; xyzs = []; lvec = []
    with open(fname, 'r') as f:
        for line in f:
            ws = line.strip().split()
            if (len(ws) > 0 and ws[0][0] != '#'):
                if (ws[0] == 'atom'):
                    xyzs.append([float(ws[1]), float(ws[2]), float(ws[3])])
                    Zs.append(elements.ELEMENT_DICT[ws[4]][0])
                elif (ws[0] == 'lattice_vector'):
                    lvec.append([float(ws[1]), float(ws[2]), float(ws[3])])
                elif (ws[0] == 'trust_radius'):
                    break
    xyzs = np.array(xyzs)
    Zs = np.array(Zs, dtype=np.int32)
    if (lvec != []):
        lvec = np.stack([[0.,0.,0.]] + lvec, axis=0)
    return xyzs, Zs, lvec

def loadPOSCAR(file_path):

    with open(file_path, 'r') as f:

        f.readline()                # Comment line
        scale = float(f.readline()) # Scaling constant

        # Lattice
        lvec = np.zeros((4, 3))
        for i in range(3):
            lvec[i+1] = np.array([float(v) for v in f.readline().strip().split()])

        # Elements
        elems = [elements.ELEMENT_DICT[e][0] for e in f.readline().strip().split()]
        Zs = []
        for e, n in zip(elems, f.readline().strip().split()):
            Zs += [e] * int(n)
        Zs = np.array(Zs, dtype=np.int32)

        # Coordinate type
        line = f.readline()
        if line[0] in 'Ss':
            line = f.readline() # Ignore optional selective dynamics line
        if line[0] in 'CcKk':
            coord_type = 'cartesian'
        else:
            coord_type = 'direct'

        # Atom coordinates
        xyzs = []
        for i in range(len(Zs)):
            xyzs.append([float(v) for v in f.readline().strip().split()[:3]])
        xyzs = np.array(xyzs)

        # Scale coordinates
        lvec *= scale
        if coord_type == 'cartesian':
            xyzs *= scale
        else:
            xyzs = np.outer(xyzs[:, 0], lvec[1]) + np.outer(xyzs[:, 1], lvec[2]) + np.outer(xyzs[:, 2], lvec[3])

    return xyzs, Zs, lvec

def writeMatrix( fout, mat ):
    for v in mat:
        for num in v: fout.write(' %f ' %num )
        fout.write('\n')

def saveGeomXSF( fname,elems,xyzs, primvec, convvec=None, bTransposed=False ):
    if convvec is None:
        primvec = convvec
    with open(fname,'w') as f:
        f.write( 'CRYSTAL\n' )
        f.write( 'PRIMVEC\n' )
        writeMatrix( f, primvec )
        f.write( 'CONVVEC\n' )
        writeMatrix( f, convvec )
        f.write( 'PRIMCOORD\n' )
        f.write( '%i %i\n' %(len(elems),1) )
        if bTransposed:
            xs = xyzs[0]
            ys = xyzs[1]
            zs = xyzs[2]
            for i in range(len(elems)):
                f.write( str(elems[i] ) ); 
                f.write( f" {xs[i]:10.10f} {ys[i]:10.10f} {zs[i]:10.10f}\n" )
        else:
            for i in range(len(elems)):
                xyzsi = xyzs[i]
                f.write( str(elems[i] ) ); 
                f.write( f" {xyzsi[0]:10.10f} {xyzsi[1]:10.10f} {xyzsi[2]:10.10f}\n" )
        f.write( '\n' )

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
        ws = f.readline().split();  
        e.append(int(ws[0])); x.append(float(ws[1])); y.append(float(ws[2])); z.append(float(ws[3])); q.append(0);
    for i in range(10000):
        line = f.readline()
        if ('BEGIN_DATAGRID_3D') in line:   
            break
        elif ('DATAGRID_3D_DENSITY') in line: 
            break
        elif ('BEGIN_DATAGRID_3D_CONTCAR_v2xsf') in line:
            break   
    ws = f.readline().split(); nDim = [int(ws[0]),int(ws[1]),int(ws[2])]
    for j in range(4):
        ws = f.readline().split(); lvec.append( [float(ws[0]),float(ws[1]),float(ws[2])] )
    f.close()
    if(verbose>0): print("nDim+1", nDim)
    nDim = (nDim[0]-1,nDim[1]-1,nDim[2]-1)
    if(verbose>0): print("lvec", lvec)
    if(verbose>0): print("reading ended")
    return [ e,x,y,z,q ], nDim, lvec

def loadNPYGeom( fname ):
    if(verbose>0): print("loading atoms")
    tmp = np.load(fname+"_atoms.npy" )
    e=tmp[0];x=tmp[1];y=tmp[2]; z=tmp[3]; q=tmp[4];
    del tmp;
    if(verbose>0): print("loading lvec")
    lvec = np.load(fname+"_vec.npy" ) 
    if(verbose>0): print("loading nDim")
    tmp = np.load(fname+"_z.npy")
    nDim = tmp.shape
    del tmp;
    if(verbose>0): print("nDim", nDim)
    if(verbose>0): print("lvec", lvec)
    if(verbose>0): print("e,x,y,z", e,x,y,z)
    return [ e,x,y,z,q ], nDim, lvec

def loadAtomsCUBE( fname ):
    bohrRadius2angstroem = 0.5291772109217 # find a good place for this
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
        x.append( (r[0] - shift[0]) * bohrRadius2angstroem )
        y.append( (r[1] - shift[1]) * bohrRadius2angstroem )
        z.append( (r[2] - shift[2]) * bohrRadius2angstroem )
        #print float(l[2])*bohrRadius2angstroem, float(l[3])*bohrRadius2angstroem, float(l[4])*bohrRadius2angstroem
        e.append( int(  l[0]) )
        q.append(0.0)
    f.close()
    return [ e,x,y,z,q ]

def primcoords2Xsf( iZs, xyzs, lvec ):
    import io as SIO
    if(verbose>0): print("lvec: ", lvec)
    sio=SIO.StringIO()
    sio.write("CRYSTAL\n")
    sio.write("PRIMVEC\n")
    sio.write("%f %f %f\n" %( lvec[1][0],lvec[1][1],lvec[1][2]) );
    sio.write("%f %f %f\n" %( lvec[2][0],lvec[2][1],lvec[2][2]) );
    sio.write("%f %f %f\n" %( lvec[3][0],lvec[3][1],lvec[3][2]) );
    sio.write("CONVVEC\n")
    sio.write("%f %f %f\n" %( lvec[1][0],lvec[1][1],lvec[1][2]) );
    sio.write("%f %f %f\n" %( lvec[2][0],lvec[2][1],lvec[2][2]) );
    sio.write("%f %f %f\n" %( lvec[3][0],lvec[3][1],lvec[3][2]) );
    sio.write("PRIMCOORD\n")
    n = len(iZs)
    sio.write("%i 1\n" %n )
    for i in range(n):
        sio.write("%i %5.6f %5.6f %5.6f\n" %(iZs[i],xyzs[0][i],xyzs[1][i],xyzs[2][i]) )
    sio.write("\n")
    sio.write("BEGIN_BLOCK_DATAGRID_3D\n");
    sio.write("some_datagrid\n");
    sio.write("BEGIN_DATAGRID_3D_whatever\n");
    s = sio.getvalue()
    #print s; exit()
    return s

def loadCellCUBE( fname ):
    bohrRadius2angstroem = 0.5291772109217 # find a good place for this
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

#	cell0 = [c0[0]*   bohrRadius2angstroem, c0[1]   *bohrRadius2angstroem, c0[2]   *bohrRadius2angstroem]
    cell0 = [0.0, 0.0, 0.0]
    cell1 = [c1[0]*n1*bohrRadius2angstroem, c1[1]*n1*bohrRadius2angstroem, c1[2]*n1*bohrRadius2angstroem]
    cell2 = [c2[0]*n2*bohrRadius2angstroem, c2[1]*n2*bohrRadius2angstroem, c2[2]*n2*bohrRadius2angstroem]
    cell3 = [c3[0]*n3*bohrRadius2angstroem, c3[1]*n3*bohrRadius2angstroem, c3[2]*n3*bohrRadius2angstroem]
    f.close()
    return [ cell0, cell1, cell2, cell3 ]

def loadNCUBE( fname ):
    bohrRadius2angstroem = 0.5291772109217 # find a good place for this
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
    return [ int(sth1[0]), int(sth2[0]), int(sth3[0]) ]

def loadGeometry(fname=None,params=None):
    if(verbose>0): print("loadGeometry ", fname)
    if fname == None:
        raise ValueError("Please provide the name of the file with coordinates")
    if params == None:
        raise ValueError("Please provide the parameters dictionary here")
    is_xyz  = fname.lower().endswith(".xyz")
    is_cube = fname.lower().endswith(".cube")
    is_xsf  = fname.lower().endswith(".xsf")
    is_npy  = fname.lower().endswith(".npy")
    if(is_xyz):
        xyzs, Zs, qs, comment = loadXYZ(fname)
        nDim = params['gridN'].copy()
        lvec = parseLvecASE(comment)
        if lvec is None:
            lvec  = np.zeros((4,3))
            lvec[ 1,:  ] = params['gridA'].copy() 
            lvec[ 2,:  ] = params['gridB'].copy()
            lvec[ 3,:  ] = params['gridC'].copy()
        atoms = [list(Zs), list(xyzs[:, 0]), list(xyzs[:, 1]), list(xyzs[:, 2]), list(qs)]
    elif(is_cube):
        atoms = loadAtomsCUBE(fname)
        lvec  = loadCellCUBE(fname)
        nDim  = loadNCUBE(fname)
    elif(is_xsf):
        atoms, nDim, lvec = loadXSFGeom( fname)
    elif(is_npy):
        atoms, nDim, lvec = loadNPYGeom( fname) # under development
        #TODO: introduce a function which reads the geometry from the .npy file
    else:
        sys.exit("ERROR!!! Unknown format of geometry system. Supported formats are: .xyz, .cube, .xsf \n\n")
    return atoms,nDim,lvec

def parseLvecASE(comment):
    '''
    Try to parse the lattice vectors in an xyz file comment line according to the extended xyz
    file format used by ASE. The origin is always at zero.

    Arguments:
        comment: str. Comment line to parse.

    Returns:
        lvec: np.array of shape (4, 3) or None. The found lattice vectors or None if the
            comment line does not match the extended xyz file format.
    '''
    match = re.match('.*Lattice=\"\s*((?:[+-]?(?:[0-9]*\.)?[0-9]+\s*){9})\"', comment)
    if match:
        lvec = np.zeros(12, dtype=np.float32)
        lvec[3:] = np.array([float(s) for s in match.group(1).split()], dtype=np.float32)
        lvec = lvec.reshape(4, 3)
    else:
        lvec = None
    return lvec

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
            bondlength=ELEMENTS[ii][6]+ELEMENTS[jj][6]
            print(" find bond ", i, j,   bondlength, r, sc, (xs[i],ys[i],zs[i]), (xs[j],ys[j],zs[j]))
            if (r<( sc * bondlength)) :
                bonds.append( (i,j) )
    return bonds

def findBondsNP( atoms, fRcut=0.7, ELEMENTS = elements.ELEMENTS ):
    bonds     = []
    bondsVecs = []
    ps     = atoms[:,1:]
    iatoms = np.arange( len(atoms), dtype=int )

    Ratoms = np.array( [ ELEMENTS[ int(ei) ][7] for ei in atoms[:,0] ] ) * frCut

    subs = []
    for i,atom in enumerate(atoms):
        p    = atom[1:]
        dp   = ps - p
        r2s  = np.sum( dp**2, axis=1 )
        mask = ( (Ratoms + Ratoms[i])**2 - r2s) > 0
        ni   = np.nonzero(mask)
        ijs  = np.empty( (ni,2), np.int32 )
        ijs[:,0] = i
        ijs[:,1] = iatoms[mask]
        subs.append( ijs  )
    bonds = np.concatenate( subs )
    return bonds    #, bondsVecs

def findBonds_( atoms, iZs, sc, ELEMENTS = elements.ELEMENTS):
    bonds = []
    n = len( atoms )
    for i in range(n):
        for j in range(i):
            d  = atoms[i]-atoms[j]
            r  = math.sqrt( np.dot(d,d) )
            ii = iZs[i]-1
            jj = iZs[j]-1
            bondlength=ELEMENTS[ii][6]+ELEMENTS[jj][6]
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
        colors.append( ELEMENTS[ FFparams[e - 1][3] -1 ][8] )
    return colors


DEFAULT_POV_HEAD_NO_CAM='''
background      { color rgb <1.0,1.0,1.0> }
//background      { color rgb <0.5,0.5,0.5> }
//global_settings { ambient_light rgb< 0.2, 0.2, 0.2> }
// ***********************************************
// macros for common shapes
// ***********************************************
#default { finish {
  ambient 0.45
  diffuse 0.84
  specular 0.22
  roughness .00001
  metallic
  phong 0.9
  phong_size 120
}
}
#macro translucentFinish(T)
 finish {
  ambient 0.45
  diffuse 0.84
  specular 0.22
  roughness .00001
  metallic 1.0
  phong 0.9
  phong_size 120
}#end
#macro a(X,Y,Z,RADIUS,R,G,B,T)
 sphere{<X,Y,Z>,RADIUS
  pigment{rgbt<R,G,B,T>}
  translucentFinish(T) 
  no_shadow  // comment this out if you want include shadows 
  }
#end
#macro b(X1,Y1,Z1,RADIUS1,X2,Y2,Z2,RADIUS2,R,G,B,T)
 cone{<X1,Y1,Z1>,RADIUS1,<X2,Y2,Z2>,RADIUS2
  pigment{rgbt<R,G,B,T>  }
  translucentFinish(T)
  no_shadow // comment this out if you want include shadows 
  }
#end
'''

DEFAULT_POV_HEAD='''
// ***********************************************
// Camera & other global settings
// ***********************************************
#declare Zoom = 30.0;
#declare Width = 800;
#declare Height = 800;
camera{
  orthographic
  location < 0,  0,  -100>
  sky      < 0, -1,    0 >
  right    < -Zoom, 0, 0>
  up       < 0, Zoom, 0 >
  look_at  < .0.0,  0.0,  0.0 >
}
'''+DEFAULT_POV_HEAD_NO_CAM



def makePovCam( pos, up=[0.0,1.0,0.0], rg=[-1.0, 0.0, 0.0], fw=[0.0, 0.0, 100.0], lpos=[0.0, 0.0,-100.0], W=10.0, H=10.0 ):
    return '''
    // ***********************************************
    // Camera & other global settings
    // ***********************************************
    #declare Zoom   = 30.0;
    #declare Width  = 800;
    #declare Height = 800;
    camera{
      orthographic
      right    %f
      up       %f
      sky      < %f, %f, %f >
      location < %f, %f, %f >
      look_at  < %f, %f, %f >
    }
    light_source    { < %f,%f,%f>  rgb <0.5,0.5,0.5> }
    ''' %(  W,H, up[0],up[1],up[2],      pos[0]-fw[0],pos[1]-fw[1],pos[2]-fw[2],    pos[0],pos[1],pos[2],    lpos[0],lpos[1],lpos[2]    )
def writePov( fname, xyzs, Zs, bonds=None, HEAD=DEFAULT_POV_HEAD, bondw=0.1, spherescale=0.25, ELEMENTS = elements.ELEMENTS ):
    fout = open( fname,"w")
    n = len(xyzs)
    fout.write( HEAD )
    for i in range(n):
        clr = ELEMENTS[Zs[i]-1][8]
        R   = ELEMENTS[Zs[i]-1][7] 
        s = 'a( %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f ) \n' %( xyzs[i][0],xyzs[i][1],xyzs[i][2], spherescale*R, clr[0]/255.0,clr[1]/255.0,clr[2]/255.0,0.0 )
        fout.write(s)
    if bonds is not None:
        for b in bonds:
            i = b[0]; j = b[1]
            clr = [128,128,128]
            s   =  'b( %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f,0.0 ) \n' %( xyzs[i][0],xyzs[i][1],xyzs[i][2], bondw, xyzs[j][0],xyzs[j][1],xyzs[j][2], bondw, clr[0]/255.0,clr[1]/255.0,clr[2]/255.0 )
            fout.write(s); 
    fout.close()

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




