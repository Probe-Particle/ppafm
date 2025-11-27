import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.append("/home/mithun/git/FireCore/")
from pyBall import atomicUtils as au
from pyBall import plotUtils   as plu
import molmesh2d as mm 

# ================== Setup

zs = np.array( [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4] ) + 1.6
#zs = np.array( [2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4] ) + 1.6

params={
    "basis"    : "cc-pvdz",
    "scf_type" : "df",
    "maxiter" : 1000,
    "step_type" : "nr",

}

samples = [
#    "OHO-h_1",
#    "HNO-h",
#    "HN-h-p",
#    "HHO-h-p_1",
#    "FAD", # Formic acid dimer
#    "C60",  # Fullerene
#    "FFPB", # 4-(4-(2,3,4,5,6- pentafluorophenylethynyl)- 2,3,5,6- tetrafluorophenylethynyl) phenylethynylbenzene
#    "Pentacene", 
#    "Phtalocyanine", 
#    "PTCDA", # Perylenetetracarboxylic dianhydride
#"H-h_1", "H-h_2", "HH-h_2", "HHH-hh", "HH-hh-p", "HH-hh", "HHH-h", "HH-p_1", "HH-pp", "HNH-h-p", "HNH-hp", "HN-hh", "HN-hp_1", "HNN-hh", "HO-h", "NHO-hh", "NN-hh", "NNN-hhh", "NNO-hh_1", "NNO-hh_2", "NO-h", "OHO-p", "O-h", "OO-h",
#"H-h_1", "H-h_2", "HH-h_1", "HH-h_2", "HHH-hh", "HH-hh-p", "HHH-h-p", "HHH-hp", "HH-hh", "HHH-h", "HH-h-p", "HH-hp", "HHH-p", "H-hh", "HHN-hh", "HHN-hp", "HHO-hh", "HHO-h-p_1", "HHO-h-p_2", "HHO-hp", "HHO-h", "HHO-p", "HH-p_1", "HH-p_2", "HH-pp", "HNH-hh", "HNH-h-p", "HNH-hp", "HN-hh", "HNH-h", "HN-hp_1", "HN-hp_2", "HN-h-p", "HNH-p", "HN-h", "HNN-hh", "HNN-hp", "HNO-hh", "HNO-h-p", "HNO-hp", "HNO-h", "HNO-p", "HN-pp", "HO-h", "H-p", "N-hh", "NHO-hh", "NHO-hp", "N-h", "NN-hh", "NN-hp", "NNN-hhh", "NNO-hh_1", "NNO-hh_2", "NN-pp", "NO-h", "NO-p", "OHO-h_1", "OHO-h_2", "OHO-h-p", "OHO-p","O-h","OO-h","O-p",

"H-p", "O-h", "O-p", "N-h","OO-h", "HN-hh", "HN-hp_1", "HN-hp_2", "HNO-h", "HNO-p", "HHO-h-p_1", "OHO-h_1", "OHO-h_2"
]

tips=[
    "H2O_H",
   "H2O_O",
   "NH3_N",
   "NH3_H",
   "HCN_H",
   "HCN_N",
   "HF_H",
   "HF_F",
   "CO_C",
   "CO_O",
   "C2H2",
]

# ================= Functions

def printPointInfo( fname, ps, ptypes, ngs ):
    np = len(ps)
    nt = len(ptypes)
    ng = len(ngs)
    if np!= nt or np!= ng: 
         print( "Error: printPointInfo() size mismatch ps(%i) ptypes(%i) ngs(%i)" %(np,nt,ng) )
    #     exit()
    n = len(ps)
    fout = open(fname,'w')
    for i in range(n):
        fout.write( ("%i %s" %(i, ptypes[i])) + str(ps[i]) + str(ngs[i])+"\n" )
    fout.close()

def dictToString( dict ):
    lst = [ "%s %s" for k,v in dict.items() ]
    return "\n".join(lst)

def toPsi4( fname, mol1, mol2, params ):
    fout = open( fname, "w" )
    #fout.write( au.psi4frags2string( mol1.enames, mol1.apos ) )
    #fout.write( au.psi4frags2string( mol1.enames, mol1.apos) )
    fout.write(
    '''memory 8GB
molecule dimer {
0 1
''')
    fout.write( "".join(mol1.toLines( )) )
    fout.write( "--\n" )
    fout.write( "".join(mol2.toLines( )) )
    fout.write(
    '''units angstrom
no_reorient
symmetry c1
}
set {
    basis ''' + params["basis"] + '''
    scf_type ''' + params["scf_type"] + '''
    maxiter ''' + str(params["maxiter"]) + '''
    step_type ''' + params["step_type"] + '''
''')
    #fout.write(  dictToString( params ) )
    fout.write('''
}
energy('b3lyp-d3', bsse_type='cp' )
gradient('b3lyp-d3', bsse_type='cp' )
''');
    fout.close()

def toXYZ( fout, mol1, mol2, comment="#comment" ):
    fout.write( "%i\n" % (len(mol1.enames)+len(mol2.enames)) )
    fout.write( "%s\n" %comment )
    mol1.toXYZ(fout)
    mol2.toXYZ(fout)

def make_scan_geoms( dirname, sample, tip, points, zs = [0.0,0.1,0.3,0.4,0.5], params=params ):
    n  = len(points)
    nz = len(zs)
    print( "make_scan_geoms ", dirname, " np ", n, " nz ", nz )
    apos0 = tip.apos.copy()
    try:
        os.mkdir( dirname )
    except:
        pass
    fxyz = open( dirname+"/movie.xyz", "w" )
    for i in range(n):
        pi = points[i]
        print( "make_scan_geoms ",i,n, dirname, "pi ", pi )
        for iz in range(nz):
            p = np.array( [ pi[0],pi[1],zs[iz] ] )
            tip.apos[:,:] = apos0[:,:] + p[None,:]
            path = dirname+"/"+("p%03i_z%02i" %(i,iz))
            os.mkdir( path )
            #toPsi4( "psi_%03i_%02i.in" %(i,iz), tip, sample, params )
            toXYZ( fxyz, tip, sample, comment="ip %3i iz %3i " %(i,iz) )
            toPsi4( path+"/psi4.in", tip, sample, params )
    fxyz.close()

def makeSamplePoints( mol, fname="test", bPlot=True, bPointInfo=True ):
    mol.findBonds()                                             #; print( "mol.bonds ", mol.bonds )
    mol.neighs()
    anew, bnew = mm.makeKinkDummy( mol.apos, mol.ngs, angMin=10.0, l=1.0 )  #;print("anew ", anew, "bnew ", bnew )  ; exit()

    bonds = [(i, j) if i < j else (j, i) for i, j in mol.bonds ]    # order bonds ( i < j )

    bonds_bak = bonds.copy()

    bonds = bonds + bnew

    nao = len(mol.apos)
    if len(anew) > 0:
        apos = np.concatenate( (mol.apos[:,:2], anew), axis=0 ) 
    else:
        apos = mol.apos[:,:2].copy()

    binds        = np.repeat(np.arange(len(bonds)), 2)  # ;print("binds ", binds )  # indexes of bonds for each point
    #bsamp       = au.makeBondSamples( bonds, apos, where=[-0.4,0.0,0.4] )
    bsamp        = au.makeBondSamples( bonds, apos, where=None )
    centers, polygons = au.colapse_to_means( bsamp, R=0.7, binds=binds )     # polygons are lists of bond indices

    #print( "mol.apos.shape, centers.shape ", mol.apos.shape, centers.shape  )
    points = np.concatenate( (apos, centers[:,:2]), axis=0 )     # points = atoms + centers

    na  = len(apos)
    nc  = len(centers)
    nac = na + nc

    polys = [  set( binds[p] for p in poly )  for poly in polygons ]       # for each polygon center list of bonds adjecent to it

    conts, cont_closed = mm.polygonContours( polys, bonds )             #;print( "conts 1 ", conts  )   # order contours ( ordered list of points for each polygon )
    cps_,cpis = mm.controusPoints( conts, points, centers, beta=0.3 )   #;print( "cps 1 ", cps  )       # orderd list of points for each contour
    cps       = np.concatenate( cps_, axis=0 )                          #;print( "cps ", cps  )

    bss = mm.contours2bonds( conts, bonds )                    #;print("bss ", bss )     # bond-centers to bonds
    bcs = au.makeBondSamples( bonds, apos, where=[0.0] )       #;print("bcs ", bcs )     # bond-centers

    bss0   = len(points)         # start bond-centers points
    cont0  = bss0 + len(bcs) 
    points = np.concatenate( (points, bcs, cps ), axis=0 )        # points = atoms + bond-centers + contours

    ptyps = mol.enames  + [ "kink" ]*len(anew) + [ "center" ]*len(conts) + ['bond']*len(bcs) + ['cp']*len(cps)
    pngs  = [ [] ]*nao  + [ [j] for i,j in bnew ]*len(anew) + conts      + bonds               + cpis

    if bPlot:
        fig = plt.figure()
        plu.plotSystem( mol, bLabels=False )
        #plt.plot( points[:,0],          points[:,1],          'ok' );    # all points
        plt.plot( points[0:na,0],       points[0:na,1],        'ok' );    # atom centers
        plt.plot( points[na:nac,0],     points[na:nac,1],      'og' );    # ring centers
        plt.plot( points[bss0:cont0,0], points[bss0:cont0,1],  'or' );    # bond centers
        plt.plot( points[cont0:,0],     points[cont0:,1],      'ob' );    # oposite to bond
        ax = plt.gca()
        for i, point in enumerate(points):  ax.annotate(str(i), (point[0], point[1]), textcoords="offset points", xytext=(5,5), ha='center')
        plt.savefig(fname+"_points.png"  )
        plt.close()

    if bPointInfo:
        printPointInfo( fname+"_point_info.txt", points, ptyps, pngs )
        #printPointInfo( dname+"_point_info.txt", points, ptyps, pngs )
    #print( "na(%i) nac(%i) bss0(%i) cont0(%i)" %( na, nac, bss0, cont0) )
    #print( "len: mol.apos(%i) anew(%i) centers(%i) bonds(%i) bcs(%i) cps(%i) cpis(%i) tot(%i)" %( len(mol.apos), len(anew), len(centers), len(bonds), len(bcs), len(cps), len(cpis), len(mol.apos)+len(anew)+len(centers)+len(bss)+len(cps) ) )

    ns = [ nao, na, nac, bss0, cont0 ]
    return points, ns

# ================== Body

base_dir = "/home/mithun/work/essential_data/AFM_SCAN/4-endgroups/"
os.makedirs(base_dir, exist_ok=True)

for isamp,sample_name in enumerate(samples):
    sample = au.AtomicSystem( "./endgroups/%s.xyz" %sample_name )
    #sample = au.AtomicSystem( "./samples/%s.xyz" %sample_name )
    points, ns = makeSamplePoints( sample, fname=sample_name )
    print( sample_name, " nPoint: ",    len(points))
    #continue
    for itip,tip_name in enumerate(tips):
        tip = au.AtomicSystem( "./Tips/%s.xyz" %tip_name )
        tip.apos[:,2] -= tip.apos[:,2].min()
        dname = os.path.join(base_dir, sample_name + "-" + tip_name)
        #make_scan_geoms( "guanine_%s" %tip_name , sample, tip, points[:ns[0],:], zs = zs, params=params )
        #make_scan_geoms( dname , sample, tip, points[:ns[0],:], zs = zs, params=params )
        make_scan_geoms( dname , sample, tip, points[:,:], zs = zs, params=params )
