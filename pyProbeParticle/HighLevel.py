#!/usr/bin/python

import os
import sys
import numpy     as np
from . import GridUtils as GU
from . import fieldFFT  as fFFT
from . import common    as PPU
from . import basUtils  as BU
from . import core
from . import cpp_utils

verbose = 1

# ===== constants 
Fmax_DEFAULT = 10.0
Vmax_DEFAULT = 10.0

# ==== PP Relaxation

def Gauss(Evib, E0, w):
    return np.exp( -0.5*((Evib - E0)/w)**2);

def symGauss( Evib, E0, w):
    return Gauss(Evib, E0, w) - Gauss(Evib, -E0, w);

def meshgrid3d(xs,ys,zs):
    Xs,Ys,Zs = np.zeros()
    Xs,Ys = np.meshgrid(xs,ys)

def trjByDir( n, d=[0.0,0.0,PPU.params['scanStep'][2]], p0=[0,0,PPU.params['scanMin'][2]] ):
    trj = np.zeros( (n,3) )
    trj[:,0] = p0[0] + (np.arange( n )[::-1])*d[0]
    trj[:,1] = p0[1] + (np.arange( n )[::-1])*d[1]
    trj[:,2] = p0[2] + (np.arange( n )[::-1])*d[2]
    return trj

def relaxedScan3D( xTips, yTips, zTips, trj=None, bF3d=False ):
    if(verbose>0): print(">>BEGIN: relaxedScan3D()")
    if(verbose>0): print(" zTips : ",zTips)
    ntips = len(zTips); 
    rTips = np.zeros((ntips,3))
    rs    = np.zeros((ntips,3))
    fs    = np.zeros((ntips,3))
    nx = len(zTips); ny = len(yTips ); nz = len(xTips);
    if( bF3d ):
        fzs    = np.zeros( ( nx,ny,nz,3) );
    else:
        fzs    = np.zeros( ( nx,ny,nz   ) );
    PPpos  = np.zeros( ( nx,ny,nz,3 ) );
    if trj is None:
        trj=np.zeros((ntips,3))
        trj[:,2]=zTips[::-1]
    for ix,x in enumerate( xTips  ):
        sys.stdout.write('\033[K')
        sys.stdout.flush()
        sys.stdout.write("\rrelax ix: {}".format(ix))
        sys.stdout.flush()
        for iy,y in enumerate( yTips  ):
            rTips[:,0] = trj[:,0] + x
            rTips[:,1] = trj[:,1] + y
            rTips[:,2] = trj[:,2]
            itrav = core.relaxTipStroke( rTips, rs, fs ) / float( len(zTips) )
            if( bF3d ):
                fzs[:,iy,ix,0] = (fs[:,0].copy()) [::-1]
                fzs[:,iy,ix,1] = (fs[:,1].copy()) [::-1]
                fzs[:,iy,ix,2] = (fs[:,2].copy()) [::-1]
            else:
                fzs[:,iy,ix] = (fs[:,2].copy()) [::-1]
            PPpos[:,iy,ix,0] = rs[::-1,0]
            PPpos[:,iy,ix,1] = rs[::-1,1]
            PPpos[:,iy,ix,2] = rs[::-1,2]
    if(verbose>0): print("<<<END: relaxedScan3D()")
    return fzs,PPpos

def perform_relaxation (lvec,FFLJ,FFel=None, FFpauli=None, FFboltz=None,FFkpfm_t0sV=None,FFkpfm_tVs0=None,tipspline=None,bPPdisp=False,bFFtotDebug=False):
    if(verbose>0): print(">>>BEGIN: perform_relaxation()")
    if tipspline is not None :
        try:
            if(verbose>0): print(" loading tip spline from "+tipspline)
            S    = np.genfromtxt(tipspline )
            xs   = S[:,0].copy();  
            if(verbose>0): print("xs: ",   xs)
            ydys = S[:,1:].copy(); 
            if(verbose>0): print("ydys: ", ydys)
            core.setTipSpline( xs, ydys )
        except:
            if(verbose>0): print("cannot load tip spline from "+tipspline)
            sys.exit()
    xTips,yTips,zTips,lvecScan = PPU.prepareScanGrids( )
    FF = FFLJ.copy()
    if ( FFel is not None):
        FF += FFel * PPU.params['charge']
        if(verbose>0): print("adding charge:", PPU.params['charge'])
    if ( FFkpfm_t0sV is not None and FFkpfm_tVs0 is not None ):
        
        FF += (np.sign(PPU.params['charge'])*FFkpfm_t0sV - FFkpfm_tVs0) * abs(PPU.params['charge']) * PPU.params['Vbias']
        if(verbose>0): print("adding charge:", PPU.params['charge'], "and bias:", PPU.params['Vbias'], "V")
    if ( FFpauli is not None ):
        FF += FFpauli * PPU.params['Apauli']
    if FFboltz != None :
        FF += FFboltz
    if bFFtotDebug:
        GU.save_vec_field( 'FFtotDebug', FF, lvec )
    core.setFF_shape( np.shape(FF), lvec )
    core.setFF_Fpointer( FF )
    if (PPU.params['stiffness'] < 0.0).any():
        PPU.params['stiffness'] = np.array([PPU.params['klat'], PPU.params['klat'], PPU.params['krad']])
    if(verbose>0): print("stiffness:", PPU.params['stiffness'])
    core.setTip(kSpring=np.array((PPU.params['stiffness'][0], PPU.params['stiffness'][1], 0.0)) / -PPU.eVA_Nm,
        kRadial=PPU.params['stiffness'][2] / -PPU.eVA_Nm)
    trj=None
    if PPU.params['tiltedScan']:
        trj = trjByDir( len(zTips), d=PPU.params['scanTilt'], p0=PPU.params['scanMin'] )
    fzs,PPpos = relaxedScan3D( xTips, yTips, zTips, trj=trj, bF3d=PPU.params['tiltedScan'] )
    if bPPdisp:
        PPdisp=PPpos.copy()
        init_pos=np.array(np.meshgrid(xTips,yTips,zTips)).transpose(3,1,2,0)+np.array([PPU.params['r0Probe'][0],PPU.params['r0Probe'][1],-PPU.params['r0Probe'][2]])
        PPdisp-=init_pos
    else:
        PPdisp = None
    if(verbose>0): print("<<<END: perform_relaxation()")
    return fzs,PPpos,PPdisp,lvecScan

# ==== Forcefield grid generation

def prepareArrays( FF, Vpot ):
    if (PPU.params["gridN"][0]<=0):
        PPU.autoGridN()
    if ( FF is None ):
        gridN = PPU.params['gridN']
        FF    = np.zeros( (gridN[2],gridN[1],gridN[0],3)    )
    else:
        PPU.params['gridN'] = np.shape( FF )
    core.setFF_Fpointer( FF )
    if ( Vpot ):
        V = np.zeros( (gridN[2],gridN[1],gridN[0])    )
        core.setFF_Epointer( V )
    else:
        V=None
    return FF, V 

def computeLJ( geomFile, speciesFile, save_format=None, computeVpot=False, Fmax=Fmax_DEFAULT, Vmax=Vmax_DEFAULT, ffModel="LJ" ):
    if(verbose>0): print(">>>BEGIN: computeLJ()")
    # --- load species (LJ potential)
    FFparams            = PPU.loadSpecies( speciesFile ) 
    elem_dict           = PPU.getFFdict(FFparams); # print elem_dict
    # --- load atomic geometry
    atoms,nDim,lvec     = BU.loadGeometry( geomFile, params=PPU.params )
    atomstring          = BU.primcoords2Xsf( PPU.atoms2iZs( atoms[0],elem_dict ), [atoms[1],atoms[2],atoms[3]], lvec );
    PPU      .params['gridN'] = nDim; PPU.params['gridA'] = lvec[1]; PPU.params['gridB'] = lvec[2]; PPU.params['gridC'] = lvec[3] # must be before parseAtoms
    if(verbose>0): print(PPU.params['gridN'],        PPU.params['gridA'],           PPU.params['gridB'],           PPU.params['gridC'])
    iZs,Rs,Qs           = PPU.parseAtoms(atoms, elem_dict, autogeom=False, PBC = PPU.params['PBC'] )
    # --- prepare LJ parameters
    iPP                 = PPU.atom2iZ( PPU.params['probeType'], elem_dict )
    # --- prepare arrays and compute
    FF,V                = prepareArrays( None, computeVpot )
    if(verbose>0): print("FFLJ.shape",FF.shape)
    core.setFF_shape( np.shape(FF), lvec )
    if ffModel=="Morse":
        REs = PPU.getAtomsRE( iPP, iZs, FFparams )
        core.getMorseFF( Rs, REs )       # THE MAIN STUFF HERE
    elif ffModel=="vdW":
        cLJs = PPU.getAtomsLJ( iPP, iZs, FFparams )
        core.getVdWFF( Rs, cLJs )       # THE MAIN STUFF HERE
    else:
        cLJs = PPU.getAtomsLJ( iPP, iZs, FFparams )
        core.getLenardJonesFF( Rs, cLJs ) # THE MAIN STUFF HERE
    # --- post porces FFs
    if Fmax is not  None:
        if(verbose>0): print("Clamp force >", Fmax)
        GU.limit_vec_field( FF, Fmax=Fmax )
    if (Vmax is not None) and computeVpot:
        if(verbose>0): print("Clamp potential >", Vmax)
        V[ V > Vmax ] =  Vmax # remove too large values
    # --- save to files ?
    if save_format is not None:
        if(verbose>0): print("computeLJ Save ", save_format) 
        GU.save_vec_field( 'FF'+ffModel, FF, lvec,  data_format=save_format, head=atomstring )
        if computeVpot:
            GU.save_scal_field( 'E'+ffModel, V, lvec,  data_format=save_format, head=atomstring )
    if(verbose>0): print("<<<END: computeLJ()")
    return FF, V, nDim, lvec

def computeELFF_pointCharge( geomFile, tip='s', save_format=None, computeVpot=False, Fmax=Fmax_DEFAULT, Vmax=Vmax_DEFAULT ):
    if(verbose>0): print(">>>BEGIN: computeELFF_pointCharge()")
    tipKinds = {'s':0,'pz':1,'dz2':2}
    tipKind  = tipKinds[tip]
    if(verbose>0): print(" ========= get electrostatic forcefiled from the point charges tip=%s %i " %(tip,tipKind))
    # --- load atomic geometry
    FFparams            = PPU.loadSpecies( ) 
    elem_dict           = PPU.getFFdict(FFparams); # print elem_dict

    atoms,nDim,lvec     = BU .loadGeometry( geomFile, params=PPU.params )
    atomstring          = BU.primcoords2Xsf( PPU.atoms2iZs( atoms[0],elem_dict ), [atoms[1],atoms[2],atoms[3]], lvec );
    iZs,Rs,Qs=PPU.parseAtoms(atoms, elem_dict=elem_dict, autogeom=False, PBC=PPU.params['PBC'] )
    # --- prepare arrays and compute
    PPU.params['gridN'] = nDim; PPU.params['gridA'] = lvec[1]; PPU.params['gridB'] = lvec[2]; PPU.params['gridC'] = lvec[3]
    if(verbose>0): print(PPU.params['gridN'], PPU.params['gridA'], PPU.params['gridB'], PPU.params['gridC'])
    FF,V = prepareArrays( None, computeVpot )
    core.setFF_shape( np.shape(FF), lvec )
    core.getCoulombFF( Rs, Qs*PPU.CoulombConst, kind=tipKind ) # THE MAIN STUFF HERE
    # --- post porces FFs
    if Fmax is not  None:
        if(verbose>0): print("Clamp force >", Fmax)
        GU.limit_vec_field( FF, Fmax=Fmax )
    if (Vmax is not None) and computeVpot:
        if(verbose>0): print("Clamp potential >", Vmax)
        V[ V > Vmax ] =  Vmax # remove too large values
    # --- save to files ?
    if save_format is not None:
        if(verbose>0): print("computeLJ Save ", save_format) 
        GU.save_vec_field( 'FFel',FF,lvec,data_format=save_format, head=atomstring )
        if computeVpot:
            GU.save_scal_field( 'Vel',V,lvec,data_format=save_format, head=atomstring )
    if(verbose>0): print("<<<END: computeELFF_pointCharge()")
    return FF, V, nDim, lvec

def computeElFF(V,lvec,nDim,tip,computeVpot=False, tilt=0.0,sigma=PPU.params['sigma'] ):
    if(verbose>0): print(" ========= get electrostatic forcefiled from hartree ")
    rho = None
    multipole = None
    if type(tip) is np.ndarray:
        rho = tip
    elif type(tip) is dict:
        multipole = tip
    else:
        if tip in {'s','px','py','pz','dx2','dy2','dz2','dxy','dxz','dyz'}:
            rho = None
            multipole={tip:1.0}
        elif tip.endswith(".xsf"):
            rho, lvec_tip, nDim_tip, tiphead = GU.loadXSF(tip)
            if any(nDim_tip != nDim):
                sys.exit("Error: Input file for tip charge density has been specified, but the dimensions are incompatible with the Hartree potential file!")
    if(verbose>0): print(" computing convolution with tip by FFT ")
    Fel_x,Fel_y,Fel_z, Vout = fFFT.potential2forces_mem( V, lvec, nDim, rho=rho, sigma=sigma, multipole = multipole, doPot=computeVpot, tilt=tilt )
    FFel = GU.packVecGrid(Fel_x,Fel_y,Fel_z)
    del Fel_x,Fel_y,Fel_z
    return FFel, Vout

def loadValenceElectronDict():
    valElDict_ = None
    namespace = {}
    try:
        fname_valelec_dict = 'valelec_dict.py'
        namespace = {}
        exec( open(fname_valelec_dict).read(), namespace )
        print("   : ", namespace['valElDict'] )
        valElDict_ = namespace['valElDict']
        print("Valence electrons loaded from local file : ", fname_valelec_dict)
    except:
        pass
    if valElDict_ is None:
        namespace = {}
        fname_valelec_dict = cpp_utils.PACKAGE_PATH+'/defaults/valelec_dict.py'
        exec(open(fname_valelec_dict).read(), namespace )
        valElDict_ = namespace['valElDict']
        print("Valence electrons loaded from default location : ", fname_valelec_dict)
    if(verbose>0): print(" Valence Electron Dict : \n", valElDict_)
    return valElDict_

def getAtomsWhichTouchPBCcell( fname, Rcut=1.0, bSaveDebug=True ):
    atoms, nDim, lvec = BU.loadGeometry( fname, params=PPU.params )
    Rs = np.array(atoms[1:4])                     # get just positions x,y,z
    inds, Rs_ = PPU.findPBCAtoms3D_cutoff( Rs, np.array(lvec[1:]), Rcut=Rcut )  # find periodic images of PBC images of atom of radius Rcut which touch our cell 
    elems = [ atoms[0][i] for i in inds ]   # atomic number of all relevant peridic images of atoms   
    if bSaveDebug:
        BU.saveGeomXSF( fname+"_TouchCell_debug.xsf",elems,Rs_, lvec[1:], convvec=lvec[1:], bTransposed=True )    # for debugging - mapping PBC images of atoms to the cell
    Rs_ = Rs_.transpose().copy()
    return Rs_, elems

def subtractCoreDensities( rho, lvec_, elems=None, Rs=None, fname=None, valElDict=None, Rcore=0.7, bSaveDebugDens=False, bSaveDebugGeom=True, head=GU.XSF_HEAD_DEFAULT ):
    lvec = lvec_[1:]
    nDim = rho.shape
    if fname is not None:
        elems,Rs = getAtomsWhichTouchPBCcell( fname, Rcut=Rcore, bSaveDebug=bSaveDebugDens )
    if valElDict is None:
        valElDict = loadValenceElectronDict()
    print("subtractCoreDensities valElDict ", valElDict)
    print("subtractCoreDensities elems ", elems)
    cRAs = np.array( [ (-valElDict[elem],Rcore) for elem in elems ] ) 
    V  = np.linalg.det( lvec )   # volume of triclinic cell
    N  = nDim[0]*nDim[1]*nDim[2]
    dV = (V/N)  # volume of one voxel
    if(verbose>0): print("V : ",V," N: ",N," dV: ", dV)  
    if(verbose>0): print("sum(RHO): ",rho.sum()," Nelec: ",rho.sum()*dV," voxel volume: ", dV)   # check sum 
    core.setFF_shape   ( rho.shape, lvec )     # set grid sampling dimension and shape
    core.setFF_Epointer( rho )                  # set pointer to array with density data (to write into)
    if(verbose>0): print(">>> Projecting Core Densities ... ")
    core.getDensityR4spline( Rs, cRAs.copy() )  # Do the job ( the Projection of atoms onto grid )
    if(verbose>0): print("sum(RHO), Nelec: ",  rho.sum(),  rho.sum()*dV)   # check sum
    if bSaveDebugDens:
        GU.saveXSF( "rho_subCoreChg.xsf", rho, lvec_, head=head )
