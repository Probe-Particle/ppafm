#!/usr/bin/python

import elements
import math
import numpy as np

verbose = 0

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
            for i in xrange(n):
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

def loadAtoms( name ):
    f = open(name,"r")
    n=0;
    l = f.readline()
    #print "--",l,"--"
    try:
        n=int(l)
    except:
                raise ValueError("First line of a xyz file should contain the number of atoms. Aborting...")
    line = f.readline() 
    if (n>0):
        n=int(l)
        e=[];x=[];y=[]; z=[]; q=[]
        i = 0;
        for line in f:
            words=line.split()
            nw = len( words)
            ie = None
            if( nw >=4 ):
                e.append( words[0] )
                x.append( float(words[1]) )
                y.append( float(words[2]) )
                z.append( float(words[3]) )
                if ( nw >=5 ):
                    q.append( float(words[4]) )
                else:
                    q.append( 0.0 )				
                i+=1
            else:
                print " skipped line : ", line
    f.close()
    return [ e,x,y,z,q ]

def loadAtomsNP(fname):
    xyzs   = [] 
    Zs     = []
    enames = []
    qs     = []
    with open(fname, 'r') as f:
        for line in f:
            wds = line.split()
            try:
                xyzs.append( ( float(wds[1]), float(wds[2]), float(wds[3]) ) )
                try:
                    iz    = int(wds[0]) 
                    Zs    .append(iz)
                    enames.append( elements.ELEMENTS[iz] )
                except:
                    ename = wds[0]
                    enames.append( ename )
                    Zs    .append( elements.ELEMENT_DICT[ename][0] )
                try:
                    q = float(wds[4])
                except:
                    q = 0
                qs.append(q)
            except:
                print "cannot interpet line: ", line
                continue
    xyzs = np.array( xyzs )
    Zs   = np.array( Zs, dtype=np.int32 )
    qs   = np.array(qs)
    return xyzs,Zs,enames,qs

def loadAtomsLines( lines ):
    xyzs   = [] 
    Zs     = []
    enames = []
    qs     = []
    for line in lines:
        wds = line.split()
        try:
            xyzs.append( ( float(wds[1]), float(wds[2]), float(wds[3]) ) )
            try:
                iz    = int(wds[0]) 
                Zs    .append(iz)
                enames.append( elements.ELEMENTS[iz] )
            except:
                ename = wds[0]
                enames.append( ename )
                Zs    .append( elements.ELEMENT_DICT[ename][0] )
            try:
                q = float(wds[4])
            except:
                q = 0
            qs.append(q)
        except:
            if(verbose>0):
                print "cannot interpet line: ", line
            continue
    xyzs = np.array( xyzs )
    Zs   = np.array( Zs, dtype=np.int32 )
    qs   = np.array(qs)
    return xyzs,Zs,enames,qs

def saveXyz(fname,elems,xyzs):
    with open(fname,'w') as f:
        n = len(elems)
        f.write( "%i\n" %n )
        f.write( "#comment\n" )
        for i in range(n):
            f.write( "%s %10.10f %10.10f %10.10f\n" %(elems[i], xyzs[i][0], xyzs[i][1], xyzs[i][2] )  )

def writeDebugXYZ( fname, lines, poss, pos0=None ):
    fout  = open(fname,"w")
    natom = int(lines[0])
    npos  = len(poss)
    if pos0 is not None:
        natom+=1
        lines.append( "U %f %f %f\n" %(pos0[0], pos0[1], pos0[2]) )
    fout.write( "%i\n" %(natom + npos) )
    fout.write( "\n" )
    for line in lines[2:natom+2]:
        fout.write( line )
    for pos in poss:
        fout.write( "He %f %f %f\n" %(pos[0], pos[1], pos[2]) )
    fout.write( "\n" )

def writeDebugXYZ_2( fname, atoms, Zs, poss, pos0 ):
    fout  = open(fname,"w")
    natom = len(atoms)
    npos  = len(poss)
    fout.write( "%i\n" %(natom + npos+1) )
    fout.write( "\n" )
    fout.write( "%i %f %f %f\n" %( 92, pos0[0], pos0[1], pos0[2] ) )
    for i in range(natom):
        fout.write( "%i %f %f %f\n" %( Zs[i],atoms[i][0], atoms[i][1], atoms[i][2]) )
    for pos in poss:
        fout.write( "He %f %f %f\n" %(pos[0], pos[1], pos[2]) )
    fout.write( "\n" )


def writeDebugXYZ__( fname, atoms, Zs ):
    fout  = open(fname,"w")
    natom = len(atoms)
    fout.write( "%i\n" %(natom ) )
    fout.write( "\n" )
    for i in range(natom):
        fout.write( "%i %f %f %f\n" %( Zs[i],atoms[i][0], atoms[i][1], atoms[i][2]) )
    fout.write( "\n" )





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
        ws = f.readline().split();  e.append(int(ws[0])); x.append(float(ws[1])); y.append(float(ws[2])); z.append(float(ws[3])); q.append(0);
    for i in range(10000):
        if ('BEGIN_DATAGRID_3D') in f.readline():   
            break
        elif ('DATAGRID_3D_DENSITY') in f.readline(): 
            break
        #elif ('BEGIN_BLOCK_DATAGRID_3D') in f.readline():
        #     break
        elif ('BEGIN_DATAGRID_3D_CONTCAR_v2xsf') in f.readline():
            break   
    ws = f.readline().split(); nDim = [int(ws[0]),int(ws[1]),int(ws[2])]
    for j in range(4):
        ws = f.readline().split(); lvec.append( [float(ws[0]),float(ws[1]),float(ws[2])] )
    f.close()
    if(verbose>0): print "nDim+1", nDim
    #nDim.reverse()
    #nDim = np.array(nDim)-1
    nDim = (nDim[0]-1,nDim[1]-1,nDim[2]-1)
    if(verbose>0): print "lvec", lvec
    #print "e,x,y,z", e,x,y,z
    if(verbose>0): print "reading ended"
    return [ e,x,y,z,q ], nDim, lvec

def loadNPYGeom( fname ):
    if(verbose>0): print "loading atoms"
    tmp = np.load(fname+"_atoms.npy" )
    e=tmp[0];x=tmp[1];y=tmp[2]; z=tmp[3]; q=tmp[4];
    del tmp;
    if(verbose>0): print "loading lvec"
    lvec = np.load(fname+"_vec.npy" ) 
    if(verbose>0): print "loading nDim"
    tmp = np.load(fname+"_z.npy")
    nDim = tmp.shape
    del tmp;
    if(verbose>0): print "nDim", nDim
    if(verbose>0): print "lvec", lvec
    if(verbose>0): print "e,x,y,z", e,x,y,z
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
        #print l
        x.append( (r[0] - shift[0]) * bohrRadius2angstroem )
        y.append( (r[1] - shift[1]) * bohrRadius2angstroem )
        z.append( (r[2] - shift[2]) * bohrRadius2angstroem )
        #print float(l[2])*bohrRadius2angstroem, float(l[3])*bohrRadius2angstroem, float(l[4])*bohrRadius2angstroem
        e.append( int(  l[0]) )
        q.append(0.0)
    f.close()
    return [ e,x,y,z,q ]

def primcoords2Xsf( iZs, xyzs, lvec ):
    import cStringIO as SIO
    #print "iZs:  ", iZs
    #print "xyzs: ", xyzs
    if(verbose>0): print "lvec: ", lvec
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
    if(verbose>0): print "loadGeometry ", fname
    if fname == None:
        raise ValueError("Please provide the name of the file with coordinates")
    if params == None:
        raise ValueError("Please provide the parameters dictionary here")
    is_xyz  = fname.lower().endswith(".xyz")
    is_cube = fname.lower().endswith(".cube")
    is_xsf  = fname.lower().endswith(".xsf")
    is_npy  = fname.lower().endswith(".npy")
    if(is_xyz):
        atoms = loadAtoms(fname)
        nDim  = params['gridN'].copy()
        lvec  = np.zeros((4,3))
        lvec[ 1,:  ] = params['gridA'].copy() 
        lvec[ 2,:  ] = params['gridB'].copy()
        lvec[ 3,:  ] = params['gridC'].copy()
    elif(is_cube):
        atoms = loadAtomsCUBE(fname)
        lvec  = loadCellCUBE(fname)
        nDim  = loadNCUBE(fname)
    elif(is_xsf):
        atoms, nDim, lvec = loadXSFGeom( fname)
    elif(is_npy):
        atoms, nDim, lvec = loadNPYGeom( fname) # under development
        #raise ValueError("reading the geometry from the .npy file is not yet implemented")
        #TODO: introduce a function which reads the geometry from the .npy file
    else:
        sys.exit("ERROR!!! Unknown format of geometry system. Supported formats are: .xyz, .cube, .xsf \n\n")
    return atoms,nDim,lvec

def findBonds( atoms, iZs, sc, ELEMENTS = elements.ELEMENTS, FFparams=None ):
    bonds = []
    xs = atoms[1]
    ys = atoms[2]
    zs = atoms[3]
    n = len( xs )
    for i in xrange(n):
        for j in xrange(i):
            dx=xs[j]-xs[i]
            dy=ys[j]-ys[i]
            dz=zs[j]-zs[i]
            r=math.sqrt( dx*dx + dy*dy + dz*dz )
            ii = iZs[i]-1
            jj = iZs[j]-1
            bondlength=ELEMENTS[ii][6]+ELEMENTS[jj][6]
            #bondlength=ELEMENTS[FFparams[ii][2]-1][6]+ELEMENTS[FFparams[jj][2]-1][6]
            print " find bond ", i, j,   bondlength, r, sc, (xs[i],ys[i],zs[i]), (xs[j],ys[j],zs[j])
            if (r<( sc * bondlength)) :
                bonds.append( (i,j) )
    return bonds


def findBonds_( atoms, iZs, sc, ELEMENTS = elements.ELEMENTS):
    bonds = []
    n = len( atoms )
    for i in xrange(n):
        for j in xrange(i):
            d  = atoms[i]-atoms[j]
            r  = math.sqrt( np.dot(d,d) )
            ii = iZs[i]-1
            jj = iZs[j]-1
            bondlength=ELEMENTS[ii][6]+ELEMENTS[jj][6]
            #bondlength=ELEMENTS[FFparams[ii][2]-1][6]+ELEMENTS[FFparams[jj][2]-1][6]
            #print " find bond ", i, j,   bondlength, r, sc, atoms[i], atoms[j]
            if (r<( sc * bondlength)) :
                bonds.append( (i,j) )
    return bonds



def findBondsSimple( xyz, rmax ):
    bonds = []
    xs = atoms[1]
    ys = atoms[2]
    zs = atoms[3]
    n = len( xs )
    for i in xrange(n):
        for j in xrange(i):
            dx=xs[j]-xs[i]
            dy=ys[j]-ys[i]
            dz=zs[j]-zs[i]
            r=math.sqrt(dx*dx+dy*dy+dz*dz)
            if (r<rmax) :
                bonds.append( (i,j) )
    return bonds

def getAtomColors( iZs, ELEMENTS = elements.ELEMENTS, FFparams=None ):
    colors=[]
    #print "getAtomColors iZs : ", iZs
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
    #''' %( rg[0],rg[1],rg[2],   up[0],up[1],up[2],  pos[0]-fw[0],pos[1]-fw[1],pos[2]-fw[2],  pos[0],pos[1],pos[2], lpos[0],lpos[1],lpos[2]  )
    #''' %(   up[0]*600,up[1]*600,up[2]*600,   rg[0]*800,rg[1]*800,rg[2]*800,   pos[0]-fw[0],pos[1]-fw[1],pos[2]-fw[2],  pos[0],pos[1],pos[2]  )


def writePov( fname, xyzs, Zs, bonds=None, HEAD=DEFAULT_POV_HEAD, bondw=0.1, spherescale=0.25, ELEMENTS = elements.ELEMENTS ):
    fout = open( fname,"w")
    n = len(xyzs)
    #print n
    fout.write( HEAD )
    for i in range(n):
        #clr = ELEMENTS[e[i]-1][8]
        clr = ELEMENTS[Zs[i]-1][8]
        R   = ELEMENTS[Zs[i]-1][7] 
        #a = (z[i] + 1.8)*2.0
        s = 'a( %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f ) \n' %( xyzs[i][0],xyzs[i][1],xyzs[i][2], spherescale*R, clr[0]/255.0,clr[1]/255.0,clr[2]/255.0,0.0 )
        #s = 'a( %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f, %10.5f ) \n' %( xyzs[i][0],xyzs[i][1],xyzs[i][2], spherescale*ELEMENTS[e[i]-1][7], clr[0]/255.0-a,clr[1]/255.0,clr[2]/255.0+a,0.0 )
        fout.write(s)
    if bonds is not None:
        #print "BONDS !!!! ", bonds
        for b in bonds:
            #print "bond ", b
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
                for i in xrange(n):
                    es[j]=xyz[0][i]
                    xs[j]=xyz[1][i] + dx
                    ys[j]=xyz[2][i] + dy
                    zs[j]=xyz[3][i] + dz
                    j+=1
    return [es,xs,ys,zs]




