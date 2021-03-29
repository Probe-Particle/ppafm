#!/usr/bin/python3 

import os
import sys
import __main__ as main
import numpy as np
#sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")


import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.fieldFFT       as fFFT
import pyProbeParticle.photo          as photo

bDebug = False

class ExcitonSystem:
    # see https://stackoverflow.com/questions/3603502/prevent-creating-new-attributes-outside-init
    __slots__ = [ 'poss','rots','Ediags','irhos','ents',    'Ham','eigEs','eigVs',    'lvecs','rhoIns','rhoCanvs','Vtip','phMaps', 'STMmap', 'wfIns'  ]
    def __init__(self):
        for at in self.__slots__:
            self.__setattr__( at, None)

# ===============================================================================================================
#      Config
# ===============================================================================================================


params={
"ydim": 500,
"xdim": 500,
"homo": "",
"lumo": "",
"dens": "",
"radius":10.0,
"ztip":6.0,
#"tip":"s",
"subsampling":6,
"excitons":False,
"volumetric":False,
"flip":False,
"save":True,
"subsys":False,
"output":"",
"cubelist":"cubefiles.ini",
"wdir":"",
"molecules":"molecules.ini",
"tipDict":"tipdict.ini",
"tipDictSTM":"tipdictstm.ini",
"images":True,
"grdebug":False,
#"current":False,
"beta":-1.0,
}

def loadParams( fname ):
    print(" >> loadParams "+fname )
    fin = open(fname,'r')
    for line in fin:
        PPU.parseParamsLine( line, params )
    #print( params )

def loadDict( fname, convertor=float ):
    # what about to use AST - https://www.kite.com/python/answers/how-to-read-a-dictionary-from-a-file-in--python#
    dct = {}
    with open(fname,'r') as fin:
        for line in fin:
            wds = line.split()
            dct[wds[0]] = convertor( wds[1] )
    return dct

def loadDicts( fname, convertor=float ):
    dcts = []
    with open(fname,'r') as fin:
        for line in fin:
            if line[0]=='#':
                continue
            wds = line.split()
            i=0
            dct={}
            nw = len(wds)
            if( nw%2!=0 ): 
                raise ValueError("odd-number of tokens on line in %s \n => cannot intertpret as (key,value) pairs ", fname )
            while(i<nw):
                dct[wds[i+0]] = convertor( wds[i+1] )
                i+=2
            dcts.append( dct )
    return dcts

# ===============================================================================================================
#      Plotting Functions
# ===============================================================================================================

def makeBox( pos, rot, a=10.0,b=20.0, byCenter=False):
    c= np.cos(rot)
    s= np.sin(rot)
    x=pos[0]; y=pos[1]
    if byCenter:
        ca=c*a*0.5;sa=s*a*0.5; 
        cb=c*b*0.5;sb=s*b*0.5; 
        xs=[x-ca+sb,x+ca+sb,x+ca-sb,x-ca-sb,x-ca+sb]
        ys=[y-sa-cb,y+sa-cb,y+sa+cb,y-sa+cb,y-sa-cb]
    else:
        xs=[x,x+c*a,x+c*a-s*b,x-s*b,x]
        ys=[y,y+s*a,y+s*a+c*b,y+c*b,y]
    return xs,ys

def plotBoxes( poss, rots, lvecs, ax=None, byCenter=False ):
    if ax is None:
        ax = plt.gca()
    #print "lvec ", lvecH
    for i in range(len(poss)):
        lvec = lvecs[i]
        #xs,ys = makeBox( poss[i], rots[i], a=lvec[2][1],b=lvec[3][2] )
        if byCenter:
            xs,ys = makeBox( poss[i], rots[i], a=float(lvec[3][2]),b=float(lvec[2][1]), byCenter=True )
        else:
            xs,ys = makeBox( poss[i], rots[i], a=float(lvec[3][2]),b=float(lvec[2][1]), byCenter=False )
        ax.plot(xs,ys,linewidth=0.5)
        ax.plot(xs[0],ys[0],'.',markersize=5)

# ================================================================
#          Sub task extracted from MAIN
# ================================================================

def loadDensityFileNames( fname ):
    fin = open( fname )
    names = [] 
    for line in fin:
        ws = line.split()
        #print( "ws ",  ws )
        if( len(ws)>1 ):
            names.append( [w.strip() for w in ws] )
        else:
            names.append( ws[0].strip() )
    #print( "cubeNames", names )
    return names

def loadMolecules( fname ):
    S = ExcitonSystem()
    DATA    = np.genfromtxt( fname, skip_header=1 )
    if (len(DATA.shape) == 1):
        DATA=np.reshape(DATA,(1,DATA.shape[0]))
    #print( "DATA.shape ", DATA.shape )
    S.poss   = DATA[:, :3]   # positions
    S.rots   = DATA[:,  3]   # rotations
    coefs    = DATA[:,4:6]   # coeficients (complex)
    S.Ediags = DATA[:,  6]   # excited state energy
    S.irhos  = ( DATA[:,7] +0.5 ).astype(np.int)  # type of transity file
    if len(DATA[0,:])>8:
        S.ents = ( DATA[:,8] +0.5 ).astype(np.int)
        #print( " DATA.oents ", oents )
    else:
        S.ents = range( len(S.Ediags) )
    coefs  =  coefs[:,0]  # TODO: for the moment we take just real part, this may change in future
    S.rots*=np.pi/180. #convert to radians
    S.eigVs = [coefs]
    #S.eigEs = []
    #system = ExcitonSystem( poss,rots,Ediags,irhos,ents,    Ham,eigEs,eigVs,  rhosIn,rhoCanv,Vtip,PhMap  ]  )
    #system0 = ExcitonSystem( poss=poss,rots=rots,Ediags=Ediags,irhos=irhos,ents=ents,    Ham=None,eigEs=None,eigVs=[coefs],  lvecs=None,rhosIns=None,rhoCanvs=None,Vtip=None,PhMaps=None  )
    return S

def makeCombination( S0, inds ):
    print( "makeCombination inds ", inds )
    S  = ExcitonSystem()
    S.ents    = photo.combine( S0.ents  ,  inds )
    S.poss    = photo.combine( S0.poss  ,  inds )
    S.rots    = photo.combine( S0.rots  ,  inds )
    #coefs  = photo.combine( S0.coefs ,  inds )
    S.Ediags  = photo.combine( S0.Ediags,  inds )
    #S.eigVs   = photo.combine( S0.eigVs ,  inds )
    S.lvecs   = photo.combine( S0.lvecs ,  inds )
    S.irhos   = photo.combine( S0.irhos,   inds )
    S.rhoIns  = photo.combine( S0.rhoIns, inds )
    #system  = ExcitonSystem( poss=poss,rots=rots,Ediags=Ediags,irhos=irhos,ents=ents,    Ham=None,eigEs=None,eigVs=eigVs,  lvecs=lvecs,rhosIns=rhosIns,rhoCanvs=None,Vti=None,PhMaps=None  )
    return S

def loadRhoTrans( cubName=None ):
    if cubName is not None:
        #print(cubName)
        #print(isinstance(cubName,str))
        if (isinstance(cubName,str)):
            rhoName = wdir+cubName
            print(( ">>> Loading Transition density from ", rhoName, " ... " ))
            rhoTrans, lvec, nDim, head = GU.loadCUBE( rhoName,trden=True)
        #else: 
        #
        #print( "cubName ",   cubName )
        #    homoName=wdir+cubName[0]
        #    lumoName=wdir+cubName[1]
        #    print(( ">>> Loading HOMO from ", homoName, " ... " ))
        #    homo, lvecH, nDimH, headH = GU.loadCUBE( homoName )
        #    print(( ">>> Loading LUMO from ", lumoName, " ... " ))
        #    lumo, lvecL, nDimL, headL = GU.loadCUBE( lumoName )
        #    lvec=lvecH; nDim=nDimH; headH=headH
        #    homo = photo.normalizeGridWf( homo )
        #    lumo = photo.normalizeGridWf( lumo )
        #    rhoTrans = homo*lumo
        #    qh = (homo**2).sum()   #; print("q(homo) ",qh)
        #    ql = (lumo**2).sum()   #; print("q(lumo) ",ql)
        
    if params["flip"]:
        print("Transposing XYZ->ZXY")
        lvec=lvec[:,[2,0,1]]
        lvec=lvec[[0,3,1,2],:]
        npnDim=np.array(nDim)
        nDim=npnDim[[2,0,1]]
        print(lvec)
        rhoTrans=(np.transpose(rhoTrans,(1,2,0))).copy()
    return rhoTrans, lvec

def loadCubeFilesINI( S0, fname_ini ):
    print("Found density list from: "+fname_ini)
    cubeNames   = loadDensityFileNames( fname_ini )
    loadedRhos  = []
    loadedLvecs = [] 
    for cubName in cubeNames:
        rh,lv   = loadRhoTrans(cubName)
        loadedRhos .append(rh)
        loadedLvecs.append(lv)
    #S0.rhoIns = [ loadedRhos [i] for i in S0.irhos ]
    #S0.lvecs  = [ loadedLvecs[i] for i in S0.irhos ]
    # ToDo : we should load set of cube files here
    return  loadedRhos, loadedLvecs 

def maxshape(arrs):
    nmaxs=np.array(len(arrs[0].shape),dtype=int)
    for arr in arrs:
        nmaxs = np.maximum( nmaxs ,arr.shape )
    return nmaxs

def lvecMax(lvecs):
    lmax=np.zeros(3)
    for i,lvec in enumerate(lvecs):
        #print("lvec[%i] " %i, lvec)
        for v in lvec:
            lmax = np.maximum( lmax, v )
    return lmax

def loadCubeFiles( S0 ):
    #if ((os.path.isfile(wdir+params["homo"]) and os.path.isfile(wdir+params["lumo"])) or os.path.isfile(wdir+params["dens"]) ):
        # ---- This is the old way, without a valid cubelist, script expects a -D or -H and -L directives
        #      oposs, orots, ocoefs, oens, oirhos, oents = makeMoleculesInline( )
    if os.path.isfile(wdir+params["dens"]):
        print("Loading densities from command-line options")
        cubName=(params["dens"])
        #else:
        #    cubName=(params["homo"],params["lumo"])
        print("CUBENAMES: ",cubName)
        rhoTrans, lvec  = loadRhoTrans(cubName)
        loadedRhos  = [rhoTrans] 
        loadedLvecs = [lvec]
        nmol = len( S0.rots )
        S0.rhoIns   = [rhoTrans]*nmol
        S0.lvecs    = [lvec]    *nmol 
    else:
        if os.path.exists(wdir+params["cubelist"]): #check for cubelist ini file
            loadedRhos, loadedLvecs = loadCubeFilesINI( S0, wdir+params["cubelist"] )
            S0.rhoIns = [ loadedRhos [i] for i in S0.irhos ]
            S0.lvecs  = [ loadedLvecs[i] for i in S0.irhos ]
        else:
            print("ERROR: This is just not going to work without any input density .cub file !")
            exit()

    return loadedRhos, loadedLvecs, lvecMax(S0.lvecs)  # maxshape(S0.rhoIns)

def runExcitationSolver( system ):
    if params["subsampling"]:
        print("Using user subsampling")
        subsamp=params["subsampling"]
        if (subsamp <= 1):
            print("adjusting the subsampling to 1")
            subsamp=1
        if (subsamp >= 10):
            print("adjusting the subsampling to 10")
            subsamp=10
    else:
        subsamp = 6  # seems sufficient to obtain 1e-3 accuracy 
    print("Subsampling: ",subsamp)
    #es,vs,H = solveExcitonSystem( rhoTrans, lvec, poss, rots, Ediag=ens, ndim=(nDim[0]//subsamp,nDim[1]//subsamp,nDim[2]//subsamp), byCenter=byCenter )
    es,vs,H = photo.solveExcitonSystem( system.rhoIns, system.lvecs, system.poss, system.rots, Ediags=system.Ediags, nSub=subsamp, byCenter=byCenter )
    system.Ham   = H
    system.eigVs = vs
    system.eigEs = es
    #result["H" ][cix]=H
    #result["Hi"][cix*nvs:cix*nvs+nvs-1]=cix
    if params["save"]:
        print("Saving Hamiltonian")
        file1 = open(fnmb+"_"+str(cix)+".ham", "w")
        file1.write(str(H)+"\n")
        file1.write(str(es)+"\n")
        file1.write(str(vs)+"\n")
        file1.close()
    return es,vs,H

def makeSTMmap_2D( S, coefs, wfTip, dd_canv, byCenter=False  ):
    Tmap, wfCanv = photo.photonMap2D_stamp( S.wfIns, S.lvecs, wfTip, dd_canv[:2], rots=S.rots, poss=S.poss, coefs=coefs, byCenter=byCenter )
    Imap = ( Tmap.real**2 + Tmap.imag**2 )
    return Imap, wfCanv

def makeSTMmap( S, coefs, wfTip, dd_canv, byCenter=False  ):
    Tmap, wfCanv = photo.photonMap3D_stamp( S.wfIns, S.lvecs, wfTip, dd_canv, rots=S.rots, poss=S.poss, coefs=coefs, byCenter=byCenter )
    Imap = ( Tmap.real**2 + Tmap.imag**2 )
    return Imap, wfCanv

def makePhotonMap( S, ipl, coefs, Vtip, dd_canv, byCenter=False, bDebugXsf=False  ):
    #print( "Volumetric ", params["volumetric"], " dd_canv ", dd_canv )
    if params["volumetric"]:
        phmap, rhoCanv_ = photo.photonMap3D_stamp( S.rhoIns, S.lvecs, Vtip, dd_canv, rots=S.rots, poss=S.poss, coefs=coefs, byCenter=byCenter )
        #phmap = np.sum(phmap_,axis=0)   # phmap_ is already 2D
        rhoCanv = np.sum(rhoCanv_  ,axis=0)
        #(dx,dy,dz)=dd
        if (bDebugXsf):
            GU.saveXSF( "rhoCanv_%03i.xsf" %ipl, rhoCanv_, dd=dd_canv )
    else:
        phmap, rhoCanv = photo.photonMap2D_stamp( S.rhoIns, S.lvecs, Vtip, dd_canv[:2], rots=S.rots, poss=S.poss, coefs=coefs, byCenter=byCenter )
        #(dx,dy)=dd
    phmap = (phmap.real**2+phmap.imag**2)
    #print( "phmap.shape ", phmap.shape  )
    S.phMaps  [ipl] = phmap
    S.rhoCanvs[ipl] = rhoCanv
    #S.rhoCanvs[0] = rhoCanv
    return rhoCanv, phmap

def makeHeader( system, sh ):
    header =str(sh[0])+' '+str(sh[1])
    header+='\n'+ str(sh[0]*dd[0]/10.)+' '+str(sh[1]*dd[1]/10.)
    header+='\nCombination(Hi): ' + str(cix)
    header+='\nEigennumber: '     + str(ipl) 
    header+='\nEnergy: '          + str(system.eigEs[ipl]) 
    header+='\nEigenvector: '     + str(system.eigVs[ipl])
    return header

def makeHeader_generic( system, sh ):
    header =str(sh[0])+' '+str(sh[1])
    header+='\n'+ str(sh[0]*dd[0]/10.)+' '+str(sh[1]*dd[1]/10.)
    return header


def imsaveTxt(fname, im, system ):
    np.savetxt(fname, im,header=makeHeader( system, im.shape ) )

def imsaveTxt_generic(fname, im, system ):
    np.savetxt(fname, im,header=makeHeader_generic( system, im.shape ) )

def plotPhotonMap( system, ipl,ncomb, nvs, byCenter=False, fname=None, dd=None ):
    rho   = system.rhoCanvs[ipl]
    #rho   = system.rhoCanvs[0]
    phMap = system.phMaps  [ipl]
    sh   = phMap.shape
    extent=( -sh[0]*dd[0]*0.5,sh[0]*dd[0]*0.5,   -sh[1]*dd[1]*0.5, sh[1]*dd[1]*0.5  )
    maxval=(np.max(rho.real))
    minval=abs(np.min(rho.real))
    maxs=np.max(np.array([maxval,minval])) #doing this to set the blue-red diverging scale white to zero in the plots

    if not params["grdebug"]:
        fig=plt.figure(figsize=(6,3))
        plt.subplot(1,2,1); plt.imshow( rho.real, extent=extent, origin='lower',cmap='seismic',vmin=-maxs,vmax=maxs);
        plt.axis('off');plt.title("E = "+("{:.1f}".format(1000*system.eigEs[ipl]) )+" meV" )
        plotBoxes( system.poss, system.rots, system.lvecs, byCenter=byCenter )
        plt.subplot(1,2,2); plt.imshow( phMap, origin='lower',extent=extent, cmap='gist_heat');
        plt.axis('off');plt.title("A = "+("{:.2e}".format(np.mean(phMap)) ))
        if params["images"]:
            print("Saving PNG image as ",fname )
            plt.savefig(fname+'.png', dpi=fig.dpi)
        plt.close()
    else:
        plt.subplot( ncomb,2*nvs,1+2*(cix*nvs+ipl)); plt.imshow( rho.real, origin='lower',extent=extent, cmap='seismic',vmin=-maxs,vmax=maxs);
        plt.axis('off');plt.title("E = "+("{:.1f}".format(1000*system.eigEs[ipl]) )+" meV" )
        plotBoxes( system.poss, system.rots, system.lvecs, byCenter=byCenter )
        plt.subplot(ncomb,2*nvs,2+2*(cix*nvs+ipl)); plt.imshow( phMap, extent=extent, origin='lower',cmap='gist_heat');
        plt.axis('off');plt.title("A = "+("{:.2e}".format(np.mean(phMap)) ))

def setPathIfExistDir( path, default=""):
    if( path != ""):
        if ( not os.path.isdir(path ) ):
            print("Specified dir does not exist! : ", path )
            quit()
        else:
            return path
    return default



def setPathIfExist( path, default=""):
    if( path != ""):
        if ( not os.path.isfile(wdir+path ) ):
            print("Specified file does not exist! : ", wdir+path )
            quit()
        else:
            return wdir+path
    return wdir+default


# ================================================================
#              MAIN
# ================================================================

if __name__ == "__main__":
    
#   import matplotlib.pyplot as plt
    from optparse import OptionParser
    np.set_printoptions(linewidth=400) #because of the implicit short line output into files


    #TODO: configuration file loading (config.ini)
    #TODO: custom tip dictionary (tipdict.ini)
    #TODO: custom coupling matrix (elements identified by their entities?) (coupling.txt)

    PARSER_DEFAULTVAL = None
    parser = OptionParser()
    parser.add_option( "-y", "--ydim",        action="store", type="int",    default=PARSER_DEFAULTVAL, help="height of canvas")
    parser.add_option( "-x", "--xdim",        action="store", type="int",    default=PARSER_DEFAULTVAL, help="width of canvas")
#    parser.add_option( "-H", "--homo",        action="store", type="string", default=PARSER_DEFAULTVAL, help="orbital of electron hole;    3D data-file (.xsf,.cube)")
#    parser.add_option( "-L", "--lumo",        action="store", type="string", default=PARSER_DEFAULTVAL, help="orbital of excited electron; 3D data-file (.xsf,.cube)")
    parser.add_option( "-D", "--dens",        action="store", type="string", default=PARSER_DEFAULTVAL, help="transition density; 3D data-file (.xsf,.cube)")
    parser.add_option( "-R", "--radius",      action="store", type="float",  default=PARSER_DEFAULTVAL, help="tip radius")
    parser.add_option( "-n", "--subsampling", action="store", type="int",    default=PARSER_DEFAULTVAL, help="subsampling for coupling calculation, recommended setting 5-10, lower is slower")
    parser.add_option( "-Z", "--ztip",        action="store", type="float",  default=PARSER_DEFAULTVAL, help="tip above substrate") #need to clarify what it exactly means
#    parser.add_option( "-t", "--tip",         action="store", type="string", default=PARSER_DEFAULTVAL, help="tip compositon s,px,py,pz,d...")
    parser.add_option( "-e", "--excitons",    action="store_true",           default=PARSER_DEFAULTVAL, help="calculate deloc. exitons of J-aggregate ( just WIP !!! )")
    parser.add_option( "-v", "--volumetric",  action="store_true",           default=PARSER_DEFAULTVAL, help="calculate on 2D grid, much faster")
    parser.add_option( "-f", "--flip",        action="store_true",           default=PARSER_DEFAULTVAL, help="transpose XYZ xsf/cube file to ZXY")
    parser.add_option( "-s", "--save",        action="store_true",           default=PARSER_DEFAULTVAL, help="save output as txt files")
    parser.add_option( "-u", "--subsys",      action="store_true",           default=PARSER_DEFAULTVAL, help="enable splitting to subsystems (EXPERIMENTAL)")
    parser.add_option( "-o", "--output",      action="store", type="string", default=PARSER_DEFAULTVAL, help="base filename for output")
#    parser.add_option( "-c", "--cubelist",    action="store", type="string", default=PARSER_DEFAULTVAL, help="read trans. density or homo/lumo using a list in a file")
    parser.add_option( "-w", "--wdir",        action="store", type="string", default=PARSER_DEFAULTVAL, help="working directory to find tr. densities and all the input files")
    parser.add_option( "-m", "--molecules",   action="store", type="string", default=PARSER_DEFAULTVAL, help="filename from which to read excitonic coordinates and other attributes")
    parser.add_option( "-i", "--images",      action="store_true",           default=PARSER_DEFAULTVAL, help="save output as images")
    parser.add_option( "-g", "--grdebug",     action="store_true",           default=PARSER_DEFAULTVAL, help="produce graphical output;")
    parser.add_option( "-b", "--beta",        action="store", type="float",  default=PARSER_DEFAULTVAL, help="tunelling current (STM) modulation beta")

    bDebugXsf = False
    #bDebugXsf = True
    #parser.add_option( "-o", "--output", action="store", type="string", default="pauli", help="output 3D data-file (.xsf)")
    (options_, args) = parser.parse_args()

    opt_dict = vars(options_)

    print( "Default values of params: \n", params )
    PPU.apply_options( opt_dict, params=params )
    

    if not params["grdebug"]:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    fnmb = "output"
    wdir = setPathIfExistDir( params["wdir"])
    print('WORKING Directory: ', wdir)
    #fnmb = setPathIfExist( params["homo"],default='output')
    #print('HOMO cube: ', params["homo"])
    #print('Default filename for output: ', fnmb)
    #setPathIfExist       ( params["lumo"])
    #print('LUMO cube: ', params["lumo"])
    if os.path.isfile( wdir+ "params.ini" ):
        print('found the params.ini, loading..')
        loadParams( wdir+"params.ini" )
    else:
        print( " there is NO params.ini => using default params dict" )

    PPU.apply_options( opt_dict, params=params )
    print( "Params after application of both inline and params.ini options: \n", params )


    if ( params["molecules"] != "molecules.ini" ):
        setPathIfExist ( params["molecules"] )
    #fnmb = fnmb
    if(not os.path.isfile(wdir+params["molecules"])):
        print("Molecular INI file does not exist in CWD: ",params["molecules"])
        print("Using default coordinates.")
        S0 = ExcitonSystem(); S0.poss=[[0.,0.,0.]]; S0.rots=[0.]; S0.Ediags=[1.]; S0.irhos=[0]; S0.ents=[0] #, S0.eigEs=[1.], S0.eigVs=[[1.]]
    else:
        print("Loading molecular parameters from: "+params["molecules"])
        S0 = loadMolecules( wdir+params["molecules"] )
        fnmb = params["molecules"]

    fnmb = setPathIfExist( params["dens"], default=fnmb)
    print("OUTPUT Basename: '"+fnmb+"'")
    _,_,lmax = loadCubeFiles( S0 )
 
    # -------- make Vtip
    hcanv = params["ydim"]
    wcanv = params["xdim"]
    #byCenter = False
    byCenter = True
    
    '''
    #tipDict =  { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
    #tipDict =  { 's': 1.0, 'py':1.0  }
    #tipDict =  { 's': 1.0, 'dy2':1.0  }
    tipDict   =  { 's': 1.0 }
    #tipDict  =  { 'px': 1.0  }
    #tipDict  =  { 'py': 1.0  }
    #tipDictSTM =  [{ 's': 1.0 }]
    #tipDictSTM =  [{ 'px': 1.0 },{ 'py': 1.0 }]
    tipDictSTM =  [{'s':.2},{ 'px': 1.0 },{ 'py': 1.0 }]
    #tipDictSTM =  [{ 'py': 1.0 }]
    #tipDictSTM =  [{ 'px': 1.0 }]
    '''

    if os.path.isfile(wdir+params["tipDict"]):
        print('Found tipDict, loading...')
        tipDict     = loadDicts( wdir + params["tipDict"] )[0]
    else:
        print("tipDict file not found, using default s-tip. (R=1.0)")
        tipDict   =  { 's': 1.0 }

    if os.path.isfile(wdir+params["tipDictSTM"]):
        print('Found tipDictSTM, loading...')
        tipDictsSTM = loadDicts( wdir + params["tipDictSTM"] )
    else:
        print("tipDictsSTM file not found, using default sp-tip.")
        tipDictsSTM =  [{'s':.2},{ 'px': 1.0 },{ 'py': 1.0 }]


    print( " tipDict     ", tipDict     )
    print( " tipDictSTM ", tipDictsSTM )

    dcanv = 0.2
    dd = (dcanv,dcanv,dcanv)
    nz_ph = int( lmax[0]/dd[0]+1 )   # because lvecs are transposed x=z
    print( "nz_ph ", nz_ph ," lmax ", lmax )
    Vtip, shifts = photo.makeTipField( (wcanv,hcanv,nz_ph), dd, z0=params["ztip"], sigma=params["radius"], multipole_dict=tipDict, b3D=params["volumetric"] )
    if bDebugXsf and params["volumetric"]:
        GU.saveXSF( wdir+"Vtip.xsf", Vtip,  dd=dd )
    if params["grdebug"]:
        if params["volumetric"]:
            fig=plt.figure(figsize=(5*2,5))
            plt.subplot(1,2,1); plt.imshow( np.fft.fftshift(Vtip[-1]) ); plt.title( 'Tip Cavity Field[Top]'    )
            plt.subplot(1,2,2); plt.imshow( np.fft.fftshift(Vtip[ 0]) ); plt.title( 'Tip Cavity Field[Bottom]' )
        else:
            fig=plt.figure(figsize=(5,5))
            plt.imshow( Vtip ); plt.title( 'Tip Cavity Field' )
    if params["beta"] > 0: # ========== STM simulation
        loadedWfs, loadedLvecs_wf = loadCubeFilesINI( S0, wdir+"wfs.ini" )
        #nmaxs_wf = maxshape(loadedWfs); print( "nmaxs_wf ", nmaxs_wf )
        lmax_wf = lvecMax( loadedLvecs_wf )
        nz_wf = int( lmax_wf[0]/dd[0] +1 )   # because lvecs are transposed x=z
        print( "nz_wf ", nz_wf ," lmax_wf ", lmax_wf )
        S0.wfIns = [ loadedWfs[i] for i in S0.irhos ]
        for i,tipDictSTM in enumerate( tipDictsSTM ):
            tipWf, shifts_STM = photo.makeTipField( (wcanv,hcanv,nz_wf), dd, z0=params["ztip"], sigma=params["radius"], multipole_dict=tipDictSTM, b3D=True, bSTM=True, beta=params["beta"] )
            STMmap_, wfCanv = makeSTMmap( S0, [ 1.0 ]*len(S0.rots), tipWf, dd, byCenter=byCenter )
            #lvecCanv = orthoLvec( tipWf.shape, dd )
            print( "tipWf.shape ", tipWf.shape, STMmap_.shape )
            # DEBUG Xsf files
            if bDebugXsf:
                GU.saveXSF( wdir+"STM_Vtip_%03i.xsf" %i, tipWf,  dd=dd )
                GU.saveXSF( wdir+"STM_wfCanv.xsf"      , wfCanv, dd=dd )
            if i==0:
                tipWf_ = tipWf**2
                STMmap = STMmap_
            else:
                tipWf_ += tipWf**2
                STMmap += STMmap_
        #if not params["hide"]:
            tipWf_ = np.fft.fftshift( tipWf_ )
            fig=plt.figure(figsize=(5*3,5))
            plt.subplot(1,3,1); plt.imshow( tipWf_.sum(axis=0) ); plt.title( 'Tip Wf'      )
            plt.subplot(1,3,2); plt.imshow( wfCanv.sum(axis=0) ); plt.title( 'Sample Wf '  )
            plt.subplot(1,3,3); plt.imshow( STMmap             ); plt.title( 'STM current ')

        S0.STMmap=STMmap
        if params["save"]:
            imsaveTxt_generic( fnmb+'_current.txt', STMmap, S0 )
            print("Saving PNG image as ",fnmb+'_current.png' )
            plt.savefig(fnmb+'_current.png', dpi=fig.dpi)
 
    #cposs,crots,ccoefs,cents,cens,combos = photo.combinator(oposs,orots,ocoefs,oents,oens)
    inds = photo.combinator(S0.ents,subsys=params["subsys"])
    print("inds type: ",type(inds))
    print( "combinator.inds ", inds )
    systems = [  makeCombination( S0, jnds ) for jnds in inds  ]

    ncomb = len(systems)
    for cix,S in enumerate( systems ):
        if params["excitons"]:
            #es,vs, H = runExcitationSolver( S.rhosIn, S.lvecs, S.poss, S.rots, S.ens )
            runExcitationSolver( S )
            nvs = len(S.eigVs)
            print("ediags",S.Ediags)
            print("eigVs",S.eigVs)
        else:
            print("ediags",S.Ediags)
            S.eigVs=[[1.]*len(S.Ediags)]
            print("eigVs",S.eigVs)
            S.eigEs=[S.Ediags[0]]
            nvs = 1
        print("nvs: ",nvs)
        if ( (cix==0) and params["grdebug"]):   # initialize figures on first combination
            fig=plt.figure(figsize=(4*nvs,2*ncomb+0.5))
            plt.tight_layout(pad=1.0)
    

        S.phMaps   = [None]*nvs
        S.rhoCanvs = [None]*nvs 
        # calculation
        print("seigvs:",S.eigVs[0])
        for ipl in range(len(S.eigVs)):
            fname=fnmb+("_%03i_%03i" %(cix, ipl) )
            makePhotonMap( S, ipl, S.eigVs[ipl], Vtip, dd, byCenter=byCenter, bDebugXsf=bDebugXsf )
            #rhoCanv, phmap = makePhotonMap( S, S.eigVs[ipl], Vtip, dd, byCenter=byCenter )
            #S.phMaps  [ipl] = phmap
            #S.rhoCanvs[ipl] = rhoCanv
            imsaveTxt( fname+'_phMap.txt'  , S.phMaps  [ipl], S )
            imsaveTxt( fname+'_rhoCanv.txt', S.rhoCanvs[ipl], S )
            #imsaveTxt( fname+'_rhoCanv.txt', S.rhoCanvs[0], S )
            plotPhotonMap( S, ipl,ncomb,nvs, byCenter=byCenter, fname=fname, dd=dd )
        if params["save"]:
            np.array( S.phMaps ).astype('float32').tofile( fnmb+"_%03i.stk" %cix )
            file1 = open(fnmb+"_%03i.hdr" %cix, "w")
            file1.write("#N_states Xdim Ydim Zdim\n")
            file1.write("# "+str(nvs)+" "+str(wcanv)+" "+str(hcanv)+"\n"+str() )
            file1.write("# H \n"); file1.write(str(S.Ham))
            file1.write("\n# EigenEnergy EigenVector \n")
            for ie,ei in enumerate( S.eigEs ):
                file1.write( str(ei)              )
                file1.write( str(S.eigVs[ie])+"\n")
            file1.close()

    
    # --- save results to file
    '''
    if params["save"]:
        print("Saving stack")
        file1 = open(fnmb+".hdr", "w")
        result["stack"].astype('float32').tofile(fnmb+'.stk') #saving stack to file for further processing
        #result["E"].astype('float32').tofile(fnmb+'.e') #saving stack to file for further processing
        file1.write("#Total_N Solver_N Xdim Ydim Zdim\n")
        file1.write("#"+str(nnn)+" "+str(nvs)+" "+str(wcanv)+" "+str(hcanv)+"\n"+str())
        file1.write("# EigenEnergy H_index EigenVector\n")
        for i in range(ncomb):
            ee = result["E" ][i]
            eev= result["Ev"][i]
            hh = result["Hi"][i]
            neev=np.shape(eev)
            file1.write(str(ee)+" ")
            file1.write(str(hh)+" ")
            for j in range(neev[0]):
                file1.write(" ")
                file1.write(str(eev[j]))
            file1.write("\n")
        file1.close()
    '''

    # --- plotting
    if params["grdebug"]:
        if params["images"]:
            print("Saving one big PNG image")
            plt.savefig(fnmb+'.png', dpi=fig.dpi)
        print("Plotting image")
        plt.show() #this is here for detaching the window from python and persist

