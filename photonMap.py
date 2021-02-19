#!/usr/bin/python3 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

'''
TODO:
 * Check units of tip size 
 * try 3D effects (not flatened z-axis)
 * complex coeficients for molecular orbital phase
 * draw position and orientaion of molecules
 * move molecules by mouse

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

# ===============================================================================================================
#      Plotting Functions
# ===============================================================================================================

def makeBox(pos, rot, a=10.0,b=20.0, byCenter=False):
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

def plotBoxes( poss, rots, lvec, ax=None, byCenter=False ):
    if ax is None:
        ax = plt.gca()
    #print "lvec ", lvecH
    for i in range(len(poss)):
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

def runExcitaionSolver( rhos, lvecs, poss, rots, Ediags ):
    if options.subsampling:
        print("using user subsampling")
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
    es,vs,H = photo.solveExcitonSystem( rhos, lvecs, poss, rots, Ediags=Ediags, nSub=subsamp, byCenter=byCenter )
    result["H" ][cix]=H
    result["Hi"][cix*nvs:cix*nvs+nvs-1]=cix
    if options.save:
        file1 = open(fnmb+"_"+str(cix)+".ham", "w")
        file1.write(str(H)+"\n")
        file1.write(str(es)+"\n")
        file1.write(str(vs)+"\n")
        file1.close()
    return es,vs,H

def storePhotonMap( res, result, ipl, es, vs ):
    sh  = res.shape
    if result is not None:
        result["stack"][cix*nvs+ipl,:,:]=res
        result["E"    ][cix*nvs+ipl    ]=es[ipl]
        result["Ev"   ][cix*nvs+ipl,:  ]=vs[ipl]
    #if (options.save):
    if fname:
        header =str(sh[0])+' '+str(sh[1])
        header+='\n'+ str(sh[0]*dd[0]/10.)+' '+str(sh[1]*dd[1]/10.)
        header+='\nCombination(Hi): ' + str(cix)
        header+='\nSpin combination: '+ str(six)
        header+='\nEigennumber: '     + str(ipl) 
        header+='\nEnergy: '          + str(es[ipl]) 
        header+='\nEigenvector: '     + str(vs[ipl])
        np.savetxt(fname+'.txt',res,header=header) 
    print("combination:"      + str(cix))
    print("exciton variation:"+ str(ipl))
    print("overall index: "   + str(1+2*(cix*nvs+ipl)))

def makePhotonMap( rhoTrans, lvec, tipDict=None, rots=None, poss=None, coefs=None, Es=None, ncanv=None, byCenter=False, fname=None  ):
    if options.volumetric:
        phmap_, Vtip_, rho_, dd = photo.photonMap3D_stamp( rhoTrans, lvec, z=options.ztip, sigma=options.radius, multipole_dict=tipDict, rots=rots, poss=poss, coefs=coefs, ncanv=ncanv(wcanv,hcanv), byCenter=byCenter )
        phmap = np.sum(phmap_,axis=0)
        Vtip  = np.sum(Vtip_ ,axis=0)
        rho   = np.sum(rho_  ,axis=0)
        #(dx,dy,dz)=dd
    else:
        phmap, Vtip, rho, dd    = photo.photonMap2D_stamp( rhoTrans, lvec, z=options.ztip, sigma=options.radius, multipole_dict=tipDict, rots=rots, poss=poss, coefs=coefs, ncanv=ncanv, byCenter=byCenter )
        #(dx,dy)=dd
    #print("dd ",  dd)
    res = (phmap.real**2+phmap.imag**2)
    return rho, res, Vtip, dd

def plotPhotonMap( rho, phmap, byCenter=False, fname=None, dd=None ):
    sh = phmap.shape
    extent=( -sh[0]*dd[0]*0.5,sh[0]*dd[0]*0.5,-sh[1]*dd[1]*0.5, sh[1]*dd[1]*0.5)
    maxval=(np.max(rho.real))
    minval=abs(np.min(rho.real))
    maxs=np.max(np.array([maxval,minval])) #doing this to set the blue-red diverging scale white to zero in the plots
    if options.hide:
        fig=plt.figure(figsize=(6,3))
        plt.subplot(1,2,1); plt.imshow( rho.real, extent=extent, origin='image',cmap='seismic',vmin=-maxs,vmax=maxs);
        plt.axis('off');plt.title("E = "+("{:.1f}".format(1000*es[ipl]) )+" meV" )
        plotBoxes( poss, rots, lvec, byCenter=byCenter )
        plt.subplot(1,2,2); plt.imshow( phmap, extent=extent, origin='image',cmap='gist_heat');
        plt.axis('off');plt.title("A = "+("{:.2e}".format(np.mean(phmap)) ))
        if options.images:
            print("Saving PNG image as ",fname )
            plt.savefig(fname+'.png', dpi=fig.dpi)
    else:
        plt.subplot(csh[0],2*nvs,1+2*(cix*nvs+ipl)); plt.imshow( rho.real, extent=extent, origin='image',cmap='seismic',vmin=-maxs,vmax=maxs);
        plt.axis('off');plt.title("E = "+("{:.1f}".format(1000*es[ipl]) )+" meV" )
        plotBoxes( poss, rots, lvec, byCenter=byCenter )
        plt.subplot(csh[0],2*nvs,2+2*(cix*nvs+ipl)); plt.imshow( res, extent=extent, origin='image',cmap='gist_heat');
        plt.axis('off');plt.title("A = "+("{:.2e}".format(np.mean(res)) ))

# ================================================================
#              MAIN
# ================================================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from optparse import OptionParser
    PARSER_DEFAULTVAL = None
    parser = OptionParser()
    parser.add_option( "-y", "--ydim",   action="store", type="int", default="500", help="height of canvas")
    parser.add_option( "-x", "--xdim",   action="store", type="int", default="500", help="width of canvas")
    parser.add_option( "-H", "--homo",   action="store", type="string", default="homo.cube", help="orbital of electron hole;    3D data-file (.xsf,.cube)")
    parser.add_option( "-L", "--lumo",   action="store", type="string", default="lumo.cube", help="orbital of excited electron; 3D data-file (.xsf,.cube)")
    parser.add_option( "-D", "--dens",   action="store", type="string", default=PARSER_DEFAULTVAL,         help="transition density; 3D data-file (.xsf,.cube)")
    parser.add_option( "-R", "--radius", action="store", type="float",  default="1.0", help="tip radius")
    parser.add_option( "-n", "--subsampling", action="store", type="int",  default="6", help="subsampling for coupling calculation, recommended setting 5-10, lower is slower")
    parser.add_option( "-Z", "--ztip",   action="store", type="float",  default="6.0", help="tip above substrate") #need to clarify what it exactly means
    parser.add_option( "-t", "--tip",    action="store", type="string", default="s",   help="tip compositon s,px,py,pz,d...")
    parser.add_option( "-e", "--excitons",   action="store_true",  default=False, help="callculate deloc. exitons of J-aggregate ( just WIP !!! )")
    parser.add_option( "-v", "--volumetric", action="store_true", default=False,  help="calculate on 2D grid, much faster")
    parser.add_option( "-f", "--flip", action="store_true", default=False,  help="transpose XYZ xsf/cube file to ZXY")
    parser.add_option( "-s", "--save", action="store_true", default=False,  help="save output as txt files")
    parser.add_option( "-o", "--output", action="store", type="string", default="",  help="filename for output")
    parser.add_option( "-c", "--config", action="store", type="string", default=PARSER_DEFAULTVAL,  help="read from config file")
    parser.add_option( "-i", "--images", action="store_true", default=False,  help="save output as images")
    parser.add_option( "-j", "--hide", action="store_true", default=False,  help="hide any graphical output; causes saved images to split into separate items")


    #parser.add_option( "-o", "--output", action="store", type="string", default="pauli", help="output 3D data-file (.xsf)")
    (options, args) = parser.parse_args()
    #rho1, lvec1, nDim1, head1 = GU.loadXSF("./pyridine/CHGCAR.xsf")
    #rho2, lvec2, nDim2, head2 = GU.loadXSF("./CO_/CHGCAR.xsf")
    np.set_printoptions(linewidth=400)

    if options.hide:
        import matplotlib
        matplotlib.use("Agg")

    hcanv = options.ydim
    wcanv = options.xdim

    if options.dens!=PARSER_DEFAULTVAL:
        print(( ">>> Loading Transition density from ", options.dens, " ... " ))
        rhoTrans, lvec, nDim, head = GU.loadCUBE( options.dens ,trden=True)

#        dV  = (lvec[1,0]*lvec[2,1]*lvec[3,2])/((nDim[0]+1)*(nDim[1]+1)*(nDim[2]+1))
#        print("*****dV:",dV)
#        rhoTrans*=(dV)
    else: 
        if os.path.exists(options.homo) and os.path.exists(options.lumo):
            print(( ">>> Loading HOMO from ", options.homo, " ... " ))
            homo, lvecH, nDimH, headH = GU.loadCUBE( options.homo )
            print(( ">>> Loading LUMO from ", options.lumo, " ... " ))
            lumo, lvecL, nDimL, headL = GU.loadCUBE( options.lumo )
            lvec=lvecH; nDim=nDimH; headH=headH

            homo = photo.normalizeGridWf( homo )
            lumo = photo.normalizeGridWf( lumo )
            rhoTrans = homo*lumo

            #rhoTrans += 1e-5 # Debugging hack
            qh = (homo**2).sum()   ; print("q(homo) ",qh)
            ql = (lumo**2).sum()   ; print("q(lumo) ",ql)
        else:
            print("Undefined densities, exiting :,(")
            quit()

    if options.flip:
        print("Transposing XYZ->ZXY")
        lvec=lvec[:,[2,0,1]]
        lvec=lvec[[0,3,1,2],:]
        npnDim=np.array(nDim)
        nDim=npnDim[[2,0,1]]
        print(lvec)
        rhoTrans=(np.transpose(rhoTrans,(1,2,0))).copy()

    #byCenter = False
    byCenter = True

    #tipDict =  { 's': 1.0, 'pz':0.1545  , 'dz2':-0.24548  }
    #tipDict =  { 's': 1.0, 'py':1.0  }
    #tipDict =  { 's': 1.0, 'dy2':1.0  }
    tipDict =  { 's': 1.0 }
    #tipDict =  { 'px': 1.0  }
    #tipDict =  { 'py': 1.0  }

    #phmap, Vtip, rho =  photonMap2D( rhoTrans, tipDict, lvec, z=0.5, sigma=0.0, multipole_dict=tipDict )

    fromDeg = np.pi/180.
    '''
    rots  =[-30.0*fromDeg,45.0*fromDeg]
    #poss =[ [10.0,5.0] ,  [10.0,10.0] ]
    poss  =[ [-5.0,10.0] ,  [5.0,-5.0] ]
    #poss =[ [0.0,10.0]  ]
    #poss =[ [200.0,50.0] ,  [50.0,50.0] ]
    #coefs=[ [1.0,0.0],      [0.0,1.0]     ]
    #coefs=[ [1.0,0.0],      [-1.0,0.0]     ]
    coefs=[ 1.0,      -1.0     ]
    #rots =[0.0]
    #poss =[ [300.0,50.0]]
    #coefs=[ [1.0,0.0]   ]
    '''

    #oposs,orots = makePreset_row  ( 2, dx=15.9, ang=0*fromDeg ) 
    #poss,rots = makePreset_cycle( 4, R=8., ang0=30*fromDeg )
    #poss,rots = makePreset_row( 5, dx=11., ang=-45.*fromDeg )
    #poss,rots = makePreset_arr1( 3,4,R=11.6 )
    
    oposs =[ [-7.5,.0,0.],[7.5,0.0,0.]  ]
    #oposs=[0,0]
    #orots=[0,0]
    orots=[20,15]

    #orots =[fromDeg*27.,fromDeg*117.,fromDeg*27,fromDeg*117.]
#    oents = [0.]
    oents = [0.,1.]  #indices of entities, they encode the cases when one molecule has more degenerate transition densities due to symmetry 
    '''
    oposs =[ [-0.0,.0,0.],[-0.0,0.0,0.]  ]

    orots =[fromDeg*0.,fromDeg*90.]
    ocoefs = np.ones(len(orots)) #complex coefficients, one for each tr density
    
    oents = [0.,0.]  #indices of entities, they encode the cases when one molecule has more degenerate transition densities due to symmetry 
    '''

    ocoefs = np.ones(len(orots)) #complex coefficients, one for each tr density
    oens   = 1.84*np.ones(len(orots)) #diagonal coefficients with the meaning of energy
    cposs,crots,ccoefs,cents,cens,combos= photo.combinator(oposs,orots,ocoefs,oents,oens)

    #print("positions ",poss)
    #print("combination ",combos)
    csh=np.shape(crots)
    

    #intended for future use

    #coefs = [[0.9,0.1]]
    six=0

    print('combination shape: ',csh)
    if options.excitons:
        nvs=csh[1]
    else:
        nvs=1
   
    nnn=csh[0]*nvs # total number of all combinations that will be calculated

    result={ #dictionary with the photon maps, eigenenergies, there is space for more, but I'm too lazy now..
        "stack" : np.zeros([nnn,wcanv,hcanv]),
        "E"     : np.zeros(nnn),
        "Ev"    : np.zeros([nnn,nvs]),
        "H"     : np.zeros([csh[0],nvs,nvs]),
        "Hi"    : np.zeros(nnn)
    }

    if not(options.hide):
        fig=plt.figure(figsize=(2*csh[0],4*nvs))
        plt.tight_layout(pad=3.0)
    if options.output:
        fnmb=options.output
    else:
        if options.dens!=PARSER_DEFAULTVAL:
            fnmb=options.dens
        else:
            fnmb=options.homo

    # ====== Loop over configurations
    for cix in range(csh[0]):
        poss =(cposs[cix]).tolist()     ; print("Positions: ",poss)   
        rots =(crots[cix]).tolist()     ; print("Rotations: ",rots)
        coefs=(ccoefs[cix])             ; print("Coefs:     ",coefs)
        ens  =(cens[cix])
        
        # --- ToDo : in future we can make rhos = [ rho1, rho2, rho3 ]
        nmol  = len( poss )
        rhos  = [rhoTrans]*nmol
        lvecs = [lvec]*nmol 
        
        vs=[coefs]
        es=1.
        if options.excitons:
            es,vs, H = runExcitaionSolver( rhos, lvecs, poss, rots, ens )
        print("variations:",len(vs),nvs)
        for ipl in range(nvs):
            coefs=vs[ipl]
            fname=fnmb+"_"+str(cix).zfill(len(str(csh[0])))+"_"+str(ipl).zfill(len(str(nvs)))
            #photonMap2D_stamp( rhoTrans, lvec, z=options.ztip, sigma=options.radius, multipole_dict=tipDict, rots=rots, poss=poss, coefs=coefs, ncanv=ncanv, byCenter=byCenter )
            rho, res, Vtip, dd = makePhotonMap( rhos, lvecs, tipDict=tipDict, rots=rots, poss=poss, coefs=coefs, ncanv=(wcanv,hcanv), byCenter=byCenter, fname=fname )
            storePhotonMap( res, result, ipl, es, vs )
            plotPhotonMap( rho, res, byCenter=byCenter, fname=fname, dd=dd )

    print("Sorting and saving stack")
    print(np.shape(result["stack"]))
    print(np.shape(result["stack"].astype('float32')))
   
    # --- sort results
    irx=np.argsort(result["E"])
    result["E"    ]=result["E"    ][irx]
    result["Ev"   ]=result["Ev"   ][irx]
    result["stack"]=result["stack"][irx]
    # --- save results to file
    if options.save:
        file1 = open(fnmb+".hdr", "w")
        result["stack"].astype('float32').tofile(fnmb+'.stk') #saving stack to file for further processing
        #result["E"].astype('float32').tofile(fnmb+'.e') #saving stack to file for further processing
        file1.write("#Total_N Solver_N Xdim Ydim\n")
        file1.write("#"+str(nnn)+" "+str(nvs)+" "+str(wcanv)+" "+str(hcanv)+"\n")
        file1.write("# EigenEnergy H_index EigenVector\n")
        for i in range(nnn):
            ee =result["E" ][i]
            eev=result["Ev"][i]
            hh =result["Hi"][i]
            neev=np.shape(eev)
            file1.write(str(ee)+" ")
            file1.write(str(hh)+" ")
            for j in range(neev[0]):
                file1.write(" ")
                file1.write(str(eev[j]))
            file1.write("\n")
        file1.close()
    # --- plotting
    if not options.hide:
        if options.images:
            print("Saving one big PNG image")
            plt.savefig(fnmb+'.png', dpi=fig.dpi)
        print("Plotting image")
        plt.show() #this is here for detaching the window from python and persist

