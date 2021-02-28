#!/usr/bin/python3 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

'''
TODO:
 * Check FFT along z-axis (should not be periodic in that direction)
 * Save results in uniform structure 

TODO large: 
 * move molecules by mouse

Done:
* Check units of tip size
* complex coeficients for molecular orbital phase
* try 3D effects (not flatened z-axis)
* draw position and orientaion of molecules
'''

import os
import sys
import __main__ as main
import numpy as np
#import GridUtils as GU
#sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")


import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.fieldFFT       as fFFT
import pyProbeParticle.photo          as photo

bDebug = False

#from namedlist import namedlist
#from collections import  namedtuple   # problem of named tupple is that it is immutable
#ExitonSystem = namedlist( 'ExitonSystem', [ 'poss','rots','Ediags','irhos','ents',    'Ham','eigEs','eigVs',  'lvecs','rhosIns','rhoCanvs','Vtip','PhMaps'  ]  )

class ExitonSystem:
    # see https://stackoverflow.com/questions/3603502/prevent-creating-new-attributes-outside-init
    __slots__ = [ 'poss','rots','Ediags','irhos','ents',    'Ham','eigEs','eigVs',    'lvecs','rhoIns','rhoCanvs','Vtip','phMaps'  ]
    def __init__(self):
        for at in self.__slots__:
            self.__setattr__( at, None)

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
    print( "cubeNames", names )
    return names

def loadMolecules( fname ):
    S = ExitonSystem()
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
    #system = ExitonSystem( poss,rots,Ediags,irhos,ents,    Ham,eigEs,eigVs,  rhosIn,rhoCanv,Vtip,PhMap  ]  )
    #system0 = ExitonSystem( poss=poss,rots=rots,Ediags=Ediags,irhos=irhos,ents=ents,    Ham=None,eigEs=None,eigVs=[coefs],  lvecs=None,rhosIns=None,rhoCanvs=None,Vtip=None,PhMaps=None  )
    return S

def makeCombination( S0, inds ):
    print( "makeCombination inds ", inds )
    S  = ExitonSystem()
    S.ents    = photo.combine( S0.ents  ,  inds )
    S.poss    = photo.combine( S0.poss  ,  inds )
    S.rots    = photo.combine( S0.rots  ,  inds )
    #coefs  = photo.combine( S0.coefs ,  inds )
    S.Ediags  = photo.combine( S0.Ediags,  inds )
    #S.eigVs   = photo.combine( S0.eigVs ,  inds )
    S.lvecs   = photo.combine( S0.lvecs ,  inds )
    S.irhos   = photo.combine( S0.irhos,   inds )
    S.rhoIns  = photo.combine( S0.rhoIns, inds )
    #system  = ExitonSystem( poss=poss,rots=rots,Ediags=Ediags,irhos=irhos,ents=ents,    Ham=None,eigEs=None,eigVs=eigVs,  lvecs=lvecs,rhosIns=rhosIns,rhoCanvs=None,Vti=None,PhMaps=None  )
    return S

def loadRhoTrans( cubName=None):
    if cubName is not None:
        print(cubName)
        print(isinstance(cubName,str))
        if (isinstance(cubName,str)):
            rhoName = cubName
            print(( ">>> Loading Transition density from ", rhoName, " ... " ))
            rhoTrans, lvec, nDim, head = GU.loadCUBE( rhoName,trden=True)
        else: 
            print( "cubName ",   cubName )
            homoName=cubName[0]
            lumoName=cubName[1]
            print(( ">>> Loading HOMO from ", homoName, " ... " ))
            homo, lvecH, nDimH, headH = GU.loadCUBE( homoName )
            print(( ">>> Loading LUMO from ", lumoName, " ... " ))
            lumo, lvecL, nDimL, headL = GU.loadCUBE( lumoName )
            lvec=lvecH; nDim=nDimH; headH=headH
            homo = photo.normalizeGridWf( homo )
            lumo = photo.normalizeGridWf( lumo )
            rhoTrans = homo*lumo
            qh = (homo**2).sum()   ; print("q(homo) ",qh)
            ql = (lumo**2).sum()   ; print("q(lumo) ",ql)
    if options.flip:
        print("Transposing XYZ->ZXY")
        lvec=lvec[:,[2,0,1]]
        lvec=lvec[[0,3,1,2],:]
        npnDim=np.array(nDim)
        nDim=npnDim[[2,0,1]]
        print(lvec)
        rhoTrans=(np.transpose(rhoTrans,(1,2,0))).copy()
    return rhoTrans, lvec

def loadCubeFiles( S0 ):
    if ((os.path.isfile(wdir+options.homo) and os.path.isfile(wdir+options.lumo)) or os.path.isfile(wdir+options.dens) ):
        # ---- This is the old way, without a valid cubelist, script expects a -D or -H and -L directives
        #      oposs, orots, ocoefs, oens, oirhos, oents = makeMoleculesInline( )
        print("Loading densities from comand-line options")
        if os.path.isfile(wdir+options.dens):
            cubName=(wdir+options.dens)
        else:
            cubName=(wdir+options.homo,wdir+options.lumo)
        print("CUBENAMES: ",cubName)
        rhoTrans, lvec  = loadRhoTrans(cubName)
        nmol   = len( S0.poss )
        S0.rhoIns = [rhoTrans]*nmol
        S0.lvecs   = [lvec]    *nmol 
        loadedRhos  = [rhoTrans] 
        loadedLvecs = [lvec]
    else:
        if os.path.exists(wdir+options.cubelist): #check for cubelist ini file
            print("Found density list from: "+wdir+options.cubelist)
            cubeNames   = loadDensityFileNames( wdir+options.cubelist )
            loadedRhos  =[]
            loadedLvecs =[] 
            for cubName in cubeNames:
                if not(isinstance(cubName,str)):
                    cubName=(wdir+cubName[0],wdir+cubName[1])
                else:
                    cubName=wdir+cubName
                rh,lv   = loadRhoTrans(cubName)
                loadedRhos .append(rh)
                loadedLvecs.append(lv)
            S0.rhoIns =[ loadedRhos [i] for i in S0.irhos ]
            S0.lvecs   =[ loadedLvecs[i] for i in S0.irhos ]
            # ToDo : we should load set of cube files here
        else:
            print("ERROR: This is just not going to work without any input density .cub file !")
            quit()
    return loadedRhos, loadedLvecs


def runExcitationSolver( system ):
    if options.subsampling:
        print("Using user subsampling")
        subsamp=options.subsampling
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
    if options.save:
        print("Saving Hamiltonian")
        file1 = open(fnmb+"_"+str(cix)+".ham", "w")
        file1.write(str(H)+"\n")
        file1.write(str(es)+"\n")
        file1.write(str(vs)+"\n")
        file1.close()
    return es,vs,H

def makePhotonMap( S, ipl, coefs, Vtip, dd_canv, byCenter=False  ):
    print( "Volumetric ", options.volumetric, " dd_canv ", dd_canv )
    if options.volumetric:
        phmap_, rhoCanv_ = photo.photonMap3D_stamp( S.rhoIns, S.lvecs, Vtip,  dd_canv, rots=S.rots, poss=S.poss, coefs=coefs, byCenter=byCenter )
        phmap = np.sum(phmap_,axis=0)
        rhoCanv = np.sum(rhoCanv_  ,axis=0)
        #(dx,dy,dz)=dd
    else:
        phmap, rhoCanv = photo.photonMap2D_stamp( S.rhoIns, S.lvecs, Vtip, dd_canv, rots=S.rots, poss=S.poss, coefs=coefs, byCenter=byCenter )
        #(dx,dy)=dd
    phmap = (phmap.real**2+phmap.imag**2)
    S.phMaps  [ipl] = phmap
    S.rhoCanvs[ipl] = rhoCanv
    return rhoCanv, phmap

def makeHeader( system, sh ):
    header =str(sh[0])+' '+str(sh[1])
    header+='\n'+ str(sh[0]*dd[0]/10.)+' '+str(sh[1]*dd[1]/10.)
    header+='\nCombination(Hi): ' + str(cix)
    header+='\nEigennumber: '     + str(ipl) 
    header+='\nEnergy: '          + str(system.eigEs[ipl]) 
    header+='\nEigenvector: '     + str(system.eigVs[ipl])
    return header

def imsaveTxt(fname, im, system ):
    np.savetxt(fname, im,header=makeHeader( system, im.shape ) )

def plotPhotonMap( system, ipl,ncomb, nvs, byCenter=False, fname=None, dd=None ):
    rho   = system.rhoCanvs[ipl]
    phMap = system.phMaps  [ipl]
    sh   = phMap.shape
    extent=( -sh[0]*dd[0]*0.5,sh[0]*dd[0]*0.5,   -sh[1]*dd[1]*0.5, sh[1]*dd[1]*0.5  )
    maxval=(np.max(rho.real))
    minval=abs(np.min(rho.real))
    maxs=np.max(np.array([maxval,minval])) #doing this to set the blue-red diverging scale white to zero in the plots
    if options.hide:
        fig=plt.figure(figsize=(6,3))
        plt.subplot(1,2,1); plt.imshow( rho.real, extent=extent, origin='image',cmap='seismic',vmin=-maxs,vmax=maxs);
        plt.axis('off');plt.title("E = "+("{:.1f}".format(1000*system.eigEs[ipl]) )+" meV" )
        plotBoxes( system.poss, system.rots, system.lvecs, byCenter=byCenter )
        plt.subplot(1,2,2); plt.imshow( phMap, extent=extent, origin='image',cmap='gist_heat');
        plt.axis('off');plt.title("A = "+("{:.2e}".format(np.mean(phMap)) ))
        if options.images:
            print("Saving PNG image as ",fname )
            plt.savefig(fname+'.png', dpi=fig.dpi)
    else:
        plt.subplot( ncomb,2*nvs,1+2*(cix*nvs+ipl)); plt.imshow( rho.real, extent=extent, origin='image',cmap='seismic',vmin=-maxs,vmax=maxs);
        plt.axis('off');plt.title("E = "+("{:.1f}".format(1000*system.eigEs[ipl]) )+" meV" )
        plotBoxes( system.poss, system.rots, system.lvecs, byCenter=byCenter )
        plt.subplot(ncomb,2*nvs,2+2*(cix*nvs+ipl)); plt.imshow( phMap, extent=extent, origin='image',cmap='gist_heat');
        plt.axis('off');plt.title("A = "+("{:.2e}".format(np.mean(phMap)) ))

# ================================================================
#              MAIN
# ================================================================

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from optparse import OptionParser
    np.set_printoptions(linewidth=400) #because of the implicit short line output into files

    PARSER_DEFAULTVAL = None
    parser = OptionParser()
    parser.add_option( "-y", "--ydim",   action="store", type="int", default="500", help="height of canvas")
    parser.add_option( "-x", "--xdim",   action="store", type="int", default="500", help="width of canvas")
    parser.add_option( "-H", "--homo",   action="store", type="string", default=PARSER_DEFAULTVAL, help="orbital of electron hole;    3D data-file (.xsf,.cube)")
    parser.add_option( "-L", "--lumo",   action="store", type="string", default=PARSER_DEFAULTVAL, help="orbital of excited electron; 3D data-file (.xsf,.cube)")
    parser.add_option( "-D", "--dens",   action="store", type="string", default=PARSER_DEFAULTVAL, help="transition density; 3D data-file (.xsf,.cube)")
    parser.add_option( "-R", "--radius", action="store", type="float",  default="1.0", help="tip radius")
    parser.add_option( "-n", "--subsampling", action="store", type="int",  default="6", help="subsampling for coupling calculation, recommended setting 5-10, lower is slower")
    parser.add_option( "-Z", "--ztip",   action="store", type="float",  default="6.0", help="tip above substrate") #need to clarify what it exactly means
    parser.add_option( "-t", "--tip",    action="store", type="string", default="s",   help="tip compositon s,px,py,pz,d...")
    parser.add_option( "-e", "--excitons",   action="store_true",  default=False, help="calculate deloc. exitons of J-aggregate ( just WIP !!! )")
    parser.add_option( "-v", "--volumetric", action="store_true", default=False,  help="calculate on 2D grid, much faster")
    parser.add_option( "-f", "--flip", action="store_true", default=False,  help="transpose XYZ xsf/cube file to ZXY")
    parser.add_option( "-s", "--save", action="store_true", default=False,  help="save output as txt files")
    parser.add_option( "-u", "--subsys", action="store_true", default=False,  help="enable splitting to subsystems (EXPERIMENTAL)")
    parser.add_option( "-o", "--output", action="store", type="string", default=PARSER_DEFAULTVAL,  help="base filename for output")
    parser.add_option( "-c", "--cubelist", action="store", type="string", default="cubefiles.ini",  help="read trans. density or homo/lumo using a list in a file")
    parser.add_option( "-w", "--wdir", action="store", type="string", default="",  help="working directory to find tr. densities and all the input files")
    parser.add_option( "-m", "--molecules", action="store", type="string", default="molecules.ini",  help="filename from which to read excitonic coordinates and other attributes")
    parser.add_option( "-i", "--images", action="store_true", default=False,  help="save output as images")
    parser.add_option( "-j", "--hide", action="store_true", default=False,  help="hide any graphical output; causes saved images to split into separate items")

    #parser.add_option( "-o", "--output", action="store", type="string", default="pauli", help="output 3D data-file (.xsf)")
    (options, args) = parser.parse_args()
    
    if options.hide:
        import matplotlib
        matplotlib.use("Agg")

    if os.path.isdir(options.wdir): #check for working dir directive
        wdir=options.wdir+"/" #adding slash for sure
    else:
        wdir="" #directory where the script runs
    print("WORKING DIRECTORY: '"+wdir+"'")

    #with all of this, invalid comand line options situations are eliminated and filename for the output set

    fnmb=wdir+"output"
    if(options.homo != PARSER_DEFAULTVAL):
        if (not os.path.isfile(wdir+options.homo)):
            print("Specfied HOMO does not exist!")
            quit()
        else:
            fnmb=wdir+options.homo
    else:
        options.homo=""

    if(options.lumo != PARSER_DEFAULTVAL):
        if (not os.path.isfile(wdir+options.lumo)):
            print("Specfied LUMO does not exist!")
            quit()
    else:
        options.lumo=""

    if bool(os.path.isfile(wdir+options.homo)) ^ bool(os.path.isfile(wdir+options.lumo)):
        print("One HOMO or LUMO has not been specified!")
        quit()

    if(options.dens != PARSER_DEFAULTVAL):
        if (not os.path.isfile(wdir+options.dens)):
            print("Specified DENSITY does not exist!")
            quit()
        else:
            fnmb=wdir+options.dens
    else:
        options.dens=""

    if(options.molecules != PARSER_DEFAULTVAL):
        if (not os.path.isfile(wdir+options.molecules)):
            print("Specified parameter INI file does not exist!")
            quit()
        else:
            fnmb=wdir+options.molecules

    if(options.cubelist != PARSER_DEFAULTVAL):
        if (not os.path.isfile(wdir+options.cubelist)):
            print("Specified densities INI file does not exist!")
            quit()

    if options.output != PARSER_DEFAULTVAL:
        if os.path.isdir(wdir+options.output) or os.path.isdir(os.path.dirname(wdir+options.output)):
            fnmb=wdir+options.output
        else:
            print("Invalid output specified")
    print("DEFAULT OUTPUT BASENAME: '"+fnmb+"'")

    if os.path.exists(wdir+options.molecules): #check molecule ini file
        print("Found parameter ini file, loading: "+wdir+options.molecules)
        fname_param=wdir+options.molecules
        S0 = loadMolecules( fname_param )
    else:
        #without a ini file, resort to simple centered settings
        print("No input file found or specified, using default center coordinates")
        #oposs=[[0.,0.,0.]];orots=[0.];ocoefs=[1.];oens=[1.];oirhos=[0];oents=[0.];
        #system0 = ExitonSystem( poss=[[0.,0.,0.]],rots=[0.],Ediags=[1.],irhos=[0],ents=[0],    Ham=None,eigEs=[1.],eigVs=[[1.]],  rhosIn=None,rhoCanv=None,Vti=None,PhMap=None  )
        S0 = ExitonSystem(); S0.poss=[[0.,0.,0.]]; S0.rots=[0.]; S0.Ediags=[1.]; S0.irhos=[0]; S0.ents=[0] #, S0.eigEs=[1.], S0.eigVs=[[1.]]

    loadCubeFiles( S0 )
 
    # -------- make Vtip
    hcanv = options.ydim
    wcanv = options.xdim
    #byCenter = False
    byCenter = True
    #tipDict =  { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
    #tipDict =  { 's': 1.0, 'py':1.0  }
    #tipDict =  { 's': 1.0, 'dy2':1.0  }
    tipDict =  { 's': 1.0 }
    #tipDict =  { 'px': 1.0  }
    #tipDict =  { 'py': 1.0  }
    print( "Volumetric ", options.volumetric )
    dcanv = 0.2
    if options.volumetric:
        dd = (dcanv,dcanv,dcanv)
        Vtip, shifts = photo.makeTipField3D( (wcanv,hcanv,10), dd, z0=options.ztip, sigma=options.radius, multipole_dict=tipDict )
    else:
        dd = (dcanv,dcanv)
        Vtip, shifts = photo.makeTipField2D( (wcanv,hcanv   ), dd, z=options.ztip,  sigma=options.radius, multipole_dict=tipDict )

    #cposs,crots,ccoefs,cents,cens,combos = photo.combinator(oposs,orots,ocoefs,oents,oens)
    inds = photo.combinator(S0.ents,subsys=options.subsys)
    print(inds)
    systems = [  makeCombination( S0, jnds ) for jnds in inds  ]

    ncomb = len(systems)
    for cix,S in enumerate( systems ):
        if options.excitons:
            #es,vs, H = runExcitationSolver( S.rhosIn, S.lvecs, S.poss, S.rots, S.ens )
            runExcitationSolver( S )

        nvs = len(S.eigVs)
        if ( (cix==0) and not options.hide):   # initialize figures on first combination
            fig=plt.figure(figsize=(4*nvs,2*ncomb+0.5))
            plt.tight_layout(pad=1.0)

        S.phMaps   = [None]*nvs
        S.rhoCanvs = [None]*nvs 
        # calculation
        for ipl in range(len(S.eigVs)):
            fname=fnmb+("_03%i_03%i" %(cix, ipl) )
            makePhotonMap( S, ipl, S.eigVs[ipl], Vtip, dd, byCenter=byCenter )
            imsaveTxt( fname+'_phMap.txt'  , S.phMaps  [ipl], S )
            imsaveTxt( fname+'_rhoCanv.txt', S.rhoCanvs[ipl], S )
            plotPhotonMap( S, ipl,ncomb,nvs, byCenter=byCenter, fname=fname, dd=dd )
        if options.save:
            np.array( S.phMaps ).astype('float32').tofile( fnmb+"_03%i.stk" %cix )
            file1 = open(fnmb+"_%03i.hdr" %cix, "w")
            file1.write("#N_states Xdim Ydim Zdim\n")
            file1.write("# "+str(nvs)+" "+str(wcanv)+" "+str(hcanv)+"\n"+str() )
            file1.write("# H \n"); file1.write(str(S.Ham))
            file1.write("# EigenEnergy EigenVector \n")
            for ie,ei in enumerate( S.eigVs ):
                file1.write( str(ei)              )
                file1.write( str(S.eigVs[ie])+"\n")
            file1.close()

    '''
    # --- save results to file
    if options.save:
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
    if not options.hide:
        if options.images:
            print("Saving one big PNG image")
            plt.savefig(fnmb+'.png', dpi=fig.dpi)
        print("Plotting image")
        plt.show() #this is here for detaching the window from python and persist
