#!/usr/bin/python

import numpy as np
import os
import sys

import cpp_utils

# ====================== constants

eVA_Nm               =  16.0217657
CoulombConst         = -14.3996448915;

# default parameters of simulation
params={
    'PBC': True,
    'nPBC' :       np.array( [      1,        1,        1 ] ),
    'gridN':       np.array( [ -1,     -1,   -1   ] ).astype(np.int),
    'gridA':       np.array( [ 12.798,  -7.3889,  0.00000 ] ),
    'gridB':       np.array( [ 12.798,   7.3889,  0.00000 ] ),
    'gridC':       np.array( [      0,        0,      5.0 ] ),
    'moleculeShift':  np.array( [  0.0,      0.0,    0.0 ] ),
    'probeType':   'O',
    'charge':      0.00,
    'useLJ':True,
    'r0Probe'  :  np.array( [ 0.00, 0.00, 4.00] ),
    'stiffness':  np.array( [ 0.5,  0.5, 20.00] ),
    'klat': 0.5,
    'krad': 20.00,
    'tip':'s',
    'sigma': 0.7,
    'scanStep': np.array( [ 0.10, 0.10, 0.10 ] ),
    'scanMin': np.array( [   0.0,     0.0,    5.0 ] ),
    'scanMax': np.array( [  20.0,    20.0,    8.0 ] ),
    'kCantilever'  :  1800.0, 
    'f0Cantilever' :  30300.0,
    'Amplitude'    :  1.0,
    'plotSliceFrom':  16,
    'plotSliceTo'  :  22,
    'plotSliceBy'  :  1,
    'imageInterpolation': 'bicubic',
    'colorscale'   : 'gray',
    'ddisp'        :  0.05,
    'aMorse'       :  -1.6,
}

# ==============================
# ============================== Pure python functions
# ==============================

def Fz2df( F, dz=0.1, k0 = params['kCantilever'], f0=params['f0Cantilever'], n=4, units=16.0217656 ):
    '''
    conversion of vertical force Fz to frequency shift 
    according to:
    Giessibl, F. J. A direct method to calculate tip-sample forces from frequency shifts in frequency-modulation atomic force microscopy Appl. Phys. Lett. 78, 123 (2001)
    oscialltion amplitude of cantilever is A = n * dz
    '''
    x  = np.linspace(-1,1,n+1)
    y  = np.sqrt(1-x*x)
    dy =  ( y[1:] - y[:-1] )/(dz*n)
    fpi    = (n-2)**2; prefactor = ( 1 + fpi*(2/np.pi) ) / (fpi+1) # correction for small n
    dFconv = -prefactor * np.apply_along_axis( lambda m: np.convolve(m, dy, mode='valid'), axis=0, arr=F )
    return dFconv*units*f0/k0

# ==============================
# ==============================  server interface file I/O
# ==============================

# overide default parameters by parameters read from a file 
def loadParams( fname ):
    print " >> OVERWRITING SETTINGS by "+fname
    fin = open(fname,'r')
    for line in fin:
        words=line.split()
        if len(words)>=2:
            key = words[0]
            if key in params:
                if key == 'stiffness': raise ValueError("Attention!!! Parameter stifness is deprecated, please define krad and klat instead")
                val = params[key]
                print key,' is class ', val.__class__
                if   isinstance( val, bool ):
                    word=words[1].strip()
                    if (word[0]=="T") or (word[0]=="t"):
                        params[key] = True
                    else:
                        params[key] = False
                    print key, params[key], ">>",word,"<<"
                elif isinstance( val, float ):
                    params[key] = float( words[1] )
                    print key, params[key], words[1]
                elif   isinstance( val, int ):
                    params[key] = int( words[1] )
                    print key, params[key], words[1]
                elif isinstance( val, str ):
                    params[key] = words[1]
                    print key, params[key], words[1]
                elif isinstance(val, np.ndarray ):
                    if val.dtype == np.float:
                        params[key] = np.array([ float(words[1]), float(words[2]), float(words[3]) ])
                        print key, params[key], words[1], words[2], words[3]
                    elif val.dtype == np.int:
                        print key
                        params[key] = np.array([ int(words[1]), int(words[2]), int(words[3]) ])
                        print key, params[key], words[1], words[2], words[3]
            else :
                raise ValueError("Parameter {} is not known".format(key))
    fin.close()
    if (params["gridN"][0]<=0):
        params["gridN"][0]=round(np.linalg.norm(params["gridA"])*10)
        params["gridN"][1]=round(np.linalg.norm(params["gridB"])*10)
        params["gridN"][2]=round(np.linalg.norm(params["gridC"])*10)

def apply_options(opt):
    print "!!!! OVERRIDE params !!!! in Apply options:"
    print opt
    for key,value in opt.iteritems():
        if opt[key] is None:
            continue
        try:
            x=params[key]     # to make sure that such a key exists in the list. If not it will be skipped
            params[key]=value
            print key,value," applied"
        except:
            pass

# load atoms species parameters form a file ( currently used to load Lenard-Jones parameters )
def loadSpecies( fname=None ):
    if fname is None:
        print "WARRNING: loadSpecies(None) => load default atomtypes.ini"
        fname=cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini'
    print " loadSpecies from ", fname
    #FFparams=np.genfromtxt(fname,dtype=[('rmin',np.float64),('epsilon',np.float64),('atom',np.int),('symbol', '|S10')],usecols=[0,1,2,3])
    FFparams=np.genfromtxt(fname,dtype=[('rmin',np.float64),('epsilon',np.float64),('alpha',np.float64),('atom',np.int),('symbol', '|S10')],usecols=(0,1,2,3,4))
    return FFparams 

def autoGeom( Rs, shiftXY=False, fitCell=False, border=3.0 ):
    '''
    set Force-Filed and Scanning supercell to fit optimally given geometry
    then shifts the geometry in the center of the supercell
    '''
    zmax=max(Rs[2]); 	Rs[2] -= zmax
    print " autoGeom substracted zmax = ",zmax
    xmin=min(Rs[0]); xmax=max(Rs[0])
    ymin=min(Rs[1]); ymax=max(Rs[1])
    if fitCell:
        params[ 'gridA'   ][0] = (xmax-xmin) + 2*border
        params[ 'gridA'   ][1] = 0
        params[ 'gridB'   ][0] = 0
        params[ 'gridB'   ][1] = (ymax-ymin) + 2*border
        params[ 'scanMin' ][0] = 0
        params[ 'scanMin' ][1] = 0
        params[ 'scanMax' ][0] = params[ 'gridA' ][0]
        params[ 'scanMax' ][1] = params[ 'gridB' ][1]
        print " autoGeom changed cell to = ", params[ 'scanMax' ]
    if shiftXY:
        dx = -0.5*(xmin+xmax) + 0.5*( params[ 'gridA' ][0] + params[ 'gridB' ][0] ); Rs[0] += dx
        dy = -0.5*(ymin+ymax) + 0.5*( params[ 'gridA' ][1] + params[ 'gridB' ][1] ); Rs[1] += dy;
        print " autoGeom moved geometry by ",dx,dy

def wrapAtomsCell( Rs, da, db, avec, bvec ):
    M    = np.array( (avec[:2],bvec[:2]) )
    invM = np.linalg.inv(M)
    print M
    print invM
    ABs = np.dot( Rs[:,:2], invM )
    print "ABs.shape", ABs.shape
    ABs[:,0] = (ABs[:,0] +10+da)%1.0
    ABs[:,1] = (ABs[:,1] +10+db)%1.0
    Rs[:,:2] = np.dot( ABs, M )   

def PBCAtoms( Zs, Rs, Qs, avec, bvec, na=None, nb=None ):
    '''
    multiply atoms of sample along supercell vectors
    the multiplied sample geometry is used for evaluation of forcefield in Periodic-boundary-Conditions ( PBC )
    '''
    Zs_ = []
    Rs_ = []
    Qs_ = []
    if na is None:
        na=params['nPBC'][0]
    if nb is None:
        nb=params['nPBC'][1]
    for i in range(-na,na+1):
        for j in range(-nb,nb+1):
            for iatom in range(len(Zs)):
                x = Rs[iatom][0] + i*avec[0] + j*bvec[0]
                y = Rs[iatom][1] + i*avec[1] + j*bvec[1]
                #if (x>xmin) and (x<xmax) and (y>ymin) and (y<ymax):
                Zs_.append( Zs[iatom]          )
                Rs_.append( (x,y,Rs[iatom][2]) )
                Qs_.append( Qs[iatom]          )
                #print "i,j,iatom,len(Rs)", i,j,iatom,len(Rs_)
    return np.array(Zs_).copy(), np.array(Rs_).copy(), np.array(Qs_).copy()	

def getFFdict( FFparams ):
    elem_dict={}
    for i,ff in enumerate(FFparams):
        print i,ff
        #elem_dict[ff[3]] = i+1
        elem_dict[ff[4]] = i+1
    #print " elem_dict ", elem_dict
    return elem_dict

def atom2iZ( atm, elem_dict ):
    try:
        return int(atm)
    except:
        try:
            return elem_dict[atm]
        except:
            raise ValueError("Did not find atomkind: {}".format(atm))

def atoms2iZs( names, elem_dict ): 
    return np.array( [atom2iZ(name,elem_dict) for name in names], dtype=np.int32 )
     

def parseAtoms( atoms, elem_dict, PBC=True, autogeom=False, lvec=None ):
    Rs = np.array([atoms[1],atoms[2],atoms[3]]); 
    if elem_dict is None:
        print "WARRNING: elem_dict is None => iZs are zero"
        iZs=np.zeros( len(atoms[0]) )
    else:
        #iZs=np.array( [atom2iZ(atm,elem_dict) for atm in atoms[0] ], dtype=np.int32 )
        iZs = atoms2iZs( atoms[0], elem_dict )
    if autogeom:
        print "WARRNING: autoGeom shifts atoms"
        autoGeom( Rs, shiftXY=True,  fitCell=True,  border=3.0 )
    Rs = np.transpose( Rs, (1,0) ).copy()
    Qs = np.array( atoms[4] )
    if PBC:
        if lvec is not None: avec=lvec[1];         bvec=lvec[2]
        else:                avec=params['gridA']; bvec=params['gridB']
        iZs,Rs,Qs = PBCAtoms( iZs, Rs, Qs, avec=avec, bvec=bvec )
    return iZs,Rs,Qs

def get_C612( i, j, FFparams ):
    '''
    compute Lenard-Jones coefitioens C6 and C12 pair of atoms i,j
    '''
#	print i, j, FFparams[i], FFparams[j]
    Rij = FFparams[i][0] + FFparams[j][0]
    Eij = np.sqrt( FFparams[i][1] * FFparams[j][1] )
    return 2*Eij*(Rij**6), Eij*(Rij**12)

def getAtomsLJ(  iZprobe, iZs,  FFparams ):
    '''
    compute Lenard-Jones coefitioens C6 and C12 for interaction between atoms in list "iZs" and probe-particle "iZprobe"
    '''
    n   = len(iZs)
    cLJs  = np.zeros((n,2))
    for i in range(n):
        cLJs[i,0],cLJs[i,1] = get_C612( iZprobe-1, iZs[i]-1, FFparams )
    return cLJs

def getAtomsREA(  iZprobe, iZs,  FFparams, alphaFac=-1.0 ):
    '''
    compute Lenard-Jones coefitioens C6 and C12 for interaction between atoms in list "iZs" and probe-particle "iZprobe"
    '''
    n   = len(iZs)
    REAs  = np.zeros( (n,4) )
    i = iZprobe-1
    for ii in range(n):
        j = iZs[ii]-1
        #print ii, i, j
        REAs[ii,0] = FFparams[i][0] + FFparams[j][0]
        REAs[ii,1] = np.sqrt( FFparams[i][1] * FFparams[j][1] )
        REAs[ii,2] = FFparams[j][2] * alphaFac
    return REAs     #np.array( REAs, dtype=np.float32 )

def getAtomsRE(  iZprobe, iZs,  FFparams ):
    n   = len(iZs)
    #REs  = np.zeros((n,2))
    #for i in range(n):
    #    REs[i,0] = FFparams[i][0] + FFparams[j][0]
    #    REs[i,1] = np.sqrt( FFparams[i][1] * FFparams[j][1] )
    Rpp = FFparams[iZprobe-1][0]
    Epp = FFparams[iZprobe-1][1]
    REs = np.array( [ ( Rpp+ FFparams[iZs[i]-1][0],  Epp * FFparams[iZs[i]-1][1] )  for i in range(n) ] )
    REs[:,1] = np.sqrt(REs[:,1])
    return REs

def getAtomsLJ_fast( iZprobe, iZs,  FFparams ):
    #Rs  = FFparams[:,0]
    #Es  = FFparams[:,1]
    #np.array( [ (FFparams[i][0],FFparams[i][1]) for i in iZs ] )
    R = np.array( [ FFparams[i-1][0] for i in iZs ] )
    E = np.array( [ FFparams[i-1][1] for i in iZs ] )
    #R   = Rs[iZs];  E   = Es[iZs]; 
    R+=FFparams[iZprobe-1][0]
    E=np.sqrt(E*FFparams[iZprobe-1][1]); 
    cLJs = np.zeros((len(E),2))
    cLJs[:,0] = E         * R6
    cLJs[:,1] = cLJs[:,0] * R6 
    return cLJs 

# ============= Hi-Level Macros

def prepareScanGrids( ):
    '''
    Defines the grid over which the tip will scan, according to scanMin, scanMax, and scanStep.
    The origin of the grid is going to be shifted (from scanMin) by the bond length between the "Probe Particle"
    and the "Apex", so that while the point of reference on the tip used to interpret scanMin  was the Apex,
    the new point of reference used in the XSF output will be the Probe Particle.
    '''
    zTips  = np.arange( params['scanMin'][2], params['scanMax'][2]+0.00001, params['scanStep'][2] )
    xTips  = np.arange( params['scanMin'][0], params['scanMax'][0]+0.00001, params['scanStep'][0] )
    yTips  = np.arange( params['scanMin'][1], params['scanMax'][1]+0.00001, params['scanStep'][1] )
    extent=( xTips[0], xTips[-1], yTips[0], yTips[-1] )
    lvecScan =np.array([
        [(params['scanMin'] + params['r0Probe'])[0],
         (params['scanMin'] + params['r0Probe'])[1],
         (params['scanMin'] - params['r0Probe'])[2] ] ,
        [        (params['scanMax']-params['scanMin'])[0],0.0,0.0],
        [0.0,    (params['scanMax']-params['scanMin'])[1],0.0    ],
        [0.0,0.0,(params['scanMax']-params['scanMin'])[2]        ]
    ]).copy() 
    return xTips,yTips,zTips,lvecScan

def lvec2params( lvec ):
    params['gridA'] = lvec[ 1,: ].copy()
    params['gridB'] = lvec[ 2,: ].copy()
    params['gridC'] = lvec[ 3,: ].copy()

def params2lvec( ):
    lvec = np.array([
        [ 0.0, 0.0, 0.0 ],
        params['gridA'],
        params['gridB'],
        params['gridC'],
    ]).copy
    return lvec
