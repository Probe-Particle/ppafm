#!/usr/bin/python

import sys
import os
import numpy as np
import time

sys.path.append("../../")
#import pyProbeParticle.basUtils as bu
#import pyProbeParticle.atomicUtils as au
#import pyProbeParticle.GLView as glv
#import pyProbeParticle.FARFF  as fff

print( "DEBUG 1 " )
import pyopencl     as cl

print( "DEBUG 2 " )

import pyProbeParticle.basUtils
import pyProbeParticle.GridUtils as GU
import pyProbeParticle.common    as PPU
import pyProbeParticle.elements
import pyProbeParticle.oclUtils     as oclu 
import pyProbeParticle.fieldOCL     as FFcl 
import pyProbeParticle.RelaxOpenCL  as oclr
import pyProbeParticle.HighLevelOCL as hl

# ==========================================================
# ==========================================================
# ====================== TEST RUN ==========================
# ==========================================================
# ==========================================================



if __name__ == "__main__":
    print( "DEBUG 1 " )

    import matplotlib as mpl;    mpl.use('Agg');
    import matplotlib.pyplot as plt
    #import argparse
    #import datetime
    import os
    #from shutil import copyfile
    import subprocess
    from optparse import OptionParser

    #import sys
    #sys.path.append("/u/21/oinonen1/unix/PPMOpenCL")
    #sys.path.append("/u/25/prokoph1/unix/git/ProbeParticleModel")
    #sys.path.append("/home/prokop/git/ProbeParticleModel")
    #from   pyProbeParticle import basUtils
    #import pyProbeParticle.common    as PPU
    #import pyProbeParticle.GridUtils as GU
    #import pyopencl as cl
    #import pyProbeParticle.HighLevelOCL as hl
    #import pyProbeParticle.GeneratorOCL_LJC as PPGen
    #import pyProbeParticle.GeneratorOCL_LJC_RR as PPGen
    #from pyProbeParticle.GeneratorOCL_LJC import Generator
    PPGen = current_module = sys.modules[__name__]

    print( "DEBUG 1 " )
    # ============ Setup Probe Particle

    batch_size = 1
    nRot           = 1
    nBestRotations = 1

    molecules = ["formic_acid"]
    #molecules = ["out3"]
    #molecules = ["out2", "out3","benzeneBrCl2"]
    #molecules = ["benzeneBrCl2"]

    parser = OptionParser()
    parser.add_option( "-Y", "--Ymode", default='D-S-H', action="store", type="string", help="tip stiffenss [N/m]" )
    (options, args) = parser.parse_args()

    print( "DEBUG 2 " )

    print("options.Ymode: ", options.Ymode)

    #rotations = PPU.genRotations( np.array([1.0,0.0,0.0]) , np.linspace(-np.pi/2,np.pi/2, nRot) )

    #rotations = PPU.sphereTangentSpace(n=nRot) # http://blog.marmakoide.org/?p=1
    #rotations  = PPU.genRotations( np.array([0.,0.,1.]), np.arange( -np.pi, np.pi, 2*np.pi/nRot ) )
    rotations = np.array( [ [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], ] )

    print( "DEBUG 3 " )

    #import os
    i_platform = 0
    env = oclu.OCLEnvironment( i_platform = i_platform )
    FFcl.init(env)
    oclr.init(env)

    '''
    bPlatformInfo = True
    if bPlatformInfo:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        print "######################################################################"
        print 
        env.printInfo()
        print 
        print "######################################################################"
        print 
        env.printPlatformInfo()
        print 
        print "######################################################################"
    '''
    print( "DEBUG 4 " )

    '''
    lvec = np.array([
        [0.0,0.0,0.0],
        [15.0,0.0,0.0],
        [0.0,15.0,0.0],
        [0.0,0.0,15.0],
    ])
    '''
    lvec=None


    #make data generator
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='HeightMap' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='ElectrostaticMap' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='Lorenzian' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='Disks' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='QDisks' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='DisksOcclusion' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='Spheres' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='Spheres' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='SphereCaps' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='D-S-H' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='xyz' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='MultiMapSpheres' )
    #data_generator = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode='SpheresType' )
    data_generator  = Generator( molecules, rotations, batch_size, pixPerAngstrome=5, Ymode=options.Ymode, lvec=lvec  )
    #data_generator  = Generator( molecules, rotations, batch_size, pixPerAngstrome=10, Ymode=options.Ymode, lvec=lvec  )

    print( "DEBUG 5 " )

    #data_generator.use_rff      = True
    #data_generator.save_rff_xyz = True

    # --- 'MultiMapSpheres' and 'SpheresType' settings
    data_generator.bOccl = 1   # switch occlusion of atoms 0=off 1=on

    # --- 'SpheresType' setting
    #data_generator.typeSelection =  [1,6,8,16]  # select atom types for each output channel
    data_generator.typeSelection =  [1,6,7,8,16,33]  # select atom types for each output channel

    # --- 'MultiMapSpheres' settings  ztop[ichan] = (R - Rmin)/Rstep
    data_generator.nChan = 5      # number of channels, resp. atom size bins
    data_generator.Rmin  = 1.4    # minimum atom radius
    data_generator.Rstep = 0.2    # size range per bin (resp. cGenerator.nextRotationnel)

    data_generator.zmin_xyz = -2.0  # max depth for visible atoGenerator.nextRotation
    data_generator.Nmax_xyz = 3    # max number of visible atomGenerator.nextRotation

    print( "DEBUG 1 " )
    #data_generator.preHeight = True

    data_generator.projector.Rpp  = -0.5
    '''
    #data_generator.projector.zmin = -3.0
    data_generator.projector.zmin  = -1.5
    data_generator.projector.dzmax = 2.0
    data_generator.projector.tgMax = 0.6
    data_generator.projector.tgWidth = 0.1
    '''

    xs = np.linspace(0.0,10.0,100)
    dx = xs[1]-xs[0];
    xs -= dx
    ys = np.exp( -5*xs )

    data_generator.projector.Rfunc   = ys.astype(np.float32)
    data_generator.projector.invStep = dx
    data_generator.projector.Rmax    = xs[-1] - 3*dx
    plt.figure()
    plt.plot(xs,data_generator.projector.Rfunc); plt.grid()
    plt.savefig( "Rfunc.png" )
    plt.close()
    

    #data_generator.rotJitter = PPU.makeRotJitter(10, 0.3)

    #data_generator.npbc = None    # switch of PeriodicBoundaryConditions


    # --- params randomization 
    data_generator.randomize_enabled    = False
    data_generator.randomize_nz         = True 
    data_generator.randomize_parameters = True
    data_generator.randomize_tip_tilt   = True
    data_generator.randomize_distance   = True
    data_generator.rndQmax     = 0.1    # charge += rndQmax * ( rand()-0.5 )  (negative is off)
    data_generator.rndRmax     = 0.2    # charge += rndRmax * ( rand()-0.5 )  (negative is off)
    data_generator.rndEmax     = 0.5    # charge *= (1 + rndEmax     * ( rand()-0.5 )) (negative is off)
    data_generator.rndAlphaMax = -0.1   # charge *= (1 + rndAlphaMax * ( rand()-0.5 )) (negative is off)
    #data_generator.modMolParams = modMolParams_def   # custom function to modify parameters

    #data_generator.debugPlots = True
    data_generator.distAbove = 7.0
    #data_generator.distAbove = 8.0
    #data_generator.distAbove = 8.5
    #data_generator.distAbove = 9.0
    #data_generator.distAbove = 8.5
    #data_generator.distAbove = 9.0
    data_generator.distAboveDelta = None  

    data_generator.bQZ = True
    #data_generator.QZs = [0.1,0.0,-0.1,0]
    #data_generator.Qs  = np.array([0,1,0,0]) * -1.0         # Monopole   Qs-1.0[e]
    #data_generator.Qs  = np.array([10,0,-10,0]) * +1.0      # Dipole     Qpz-1.0[e*A]
    data_generator.Qs  = np.array([100,-200,100,0]) * -0.2   # Quadrupole Qdz2-1.0[e*A^2]

    #data_generator.maxTilt0 = 0.0    # symmetric tip
    data_generator.maxTilt0 = 0.5     # asymetric tip  tilted max +/-1.0 Angstreom in random direction
    #data_generator.maxTilt0 = 2.0     # asymetric tip  tilted max +/-1.0 Angstreom in random direction

    data_generator.shuffle_rotations = False
    data_generator.shuffle_molecules = False
    data_generator.nBestRotations    = nBestRotations

    # molecule are read from filename =  preName + molecules[imol] + postName
    data_generator.preName    = ""           # appended befroe path filename
    data_generator.postName   = "/pos.xyz"

    #data_generator.debugPlots = True   # plotting dubug images? much slower when True
    #data_generator.Q = -0.5;
    #data_generator.Q = -0.3;
    data_generator.Q = 0.0

    # z-weight exp(-wz*z)
    data_generator.wz      = 1.0    # deacay
    data_generator.zWeight =  data_generator.getZWeights();

    dz=0.1

    # weight-function for Fz -> df conversion ( oscilation amplitude 1.0Angstroem = 10 * 0.1 ( 10 n steps, dz=0.1 Angstroem step lenght ) )
    dfWeight = PPU.getDfWeight( 10, dz=dz )[0].astype(np.float32)
    #dfWeight, xs = PPU.getDfWeight( 10, dz=dz )
    #print " xs ", xs
    data_generator.dfWeight = dfWeight

    # plot zWeights
    plt.figure()
    plt.plot(data_generator.zWeight, '.-');
    plt.grid()
    plt.savefig( "zWeights.png" )
    plt.close()

    # plot dfWeights
    plt.figure()
    plt.plot( np.arange(len(data_generator.dfWeight))*dz , data_generator.dfWeight, '.-');
    plt.grid()
    plt.savefig( "dfWeights.png" )
    plt.close()
    #plt.show()

    # print
    #data_generator.bDfPerMol = True
    #data_generator.nDfMin    = 5
    #data_generator.nDfMax    = 15

    data_generator.bSaveFF = False
    data_generator.bMergeConv = True

    #data_generator.scan_dim = ( 100, 100, 20)
    #data_generator.scan_dim = ( 128, 128, 30)
    data_generator.scan_dim   = ( 256, 256, 30)
    data_generator.scan_start = (-12.5,-12.5) 
    data_generator.scan_end   = ( 12.5, 12.5)

    bRunTime      = True
    FFcl.bRuntime = True

    data_generator            .verbose  = 1
    data_generator.forcefield .verbose  = 1
    data_generator.scanner    .verbose  = 1

    data_generator.initFF()

    # generate 10 batches
    for i in range(9):

        print("#### generate ", i) 
        t1 = time.clock()
        Xs,Ys = data_generator[i]
        print("runTime(data_generator.next()) [s] : ", time.clock() - t1)
        
        #continue

        #Xs=Xs[::2]; Ys=Ys[::2]

        '''
        print "Ys.shape ", Ys.shape

        for i in range( Ys[0].shape[2] ):
            plt.figure()
            plt.imshow( Ys[0][:,:,i] )
            plt.title( "img[%i]" %i )

        plt.show()
        '''
        #exit()

        #print "_0"
        
        #data_generator.debugPlotSlices = range(0,Xs[0].shape[2],2)
        data_generator.debugPlotSlices = list(range(0,Xs[0].shape[2],1))

        for j in range( len(Xs) ):
            #print "_1"
            #print "j ", j
            #np.save( "X_i%03i_j%03i.npy" %(i,j), Xs[j] )
            #np.save( "Y_i%03i_j%03i.npy" %(i,j), Ys[j] )
            #print "Ys[j].shape", Ys[j].shape
            fname = "batch_%03i_%03i_" %(i,j)

            #for ichan in range( Ys[j].shape[2] ):
            #    plt.figure()
            #    plt.imshow( Ys[j][:,:,ichan] )
            #    plt.title( "i %i j %i ichan %i" %(i,j,ichan) )

            #nch = Ys[j].shape[2]
            #plt.figure(figsize=(5*nch,5))
            #for ichan in range( nch ):
            #    plt.subplot(  1, nch, ichan+1 )
            #    plt.imshow( Ys[j][:,:,ichan] )
            #    plt.title( "i %i j %i ichan %i" %(i,j,ichan) )

            print(" Ys[j].shape",  Ys[j].shape)

            np.save(  "./"+molecules[data_generator.imol]+"/Atoms.npy", Ys[j][:,:,0] )
            np.save(  "./"+molecules[data_generator.imol]+"/Bonds.npy", Ys[j][:,:,1] )

            #continue

            #data_generator.plot( "/"+fname, molecules[i*batch_size+j], X=Xs[j], Y=Ys[j], entropy=0.0, bXYZ=True )
            #data_generator.plot( "/"+fname, molecules[data_generator.imol], X=Xs[j], Y=Ys[j], entropy=0.0, bXYZ=True, bGroups=True )
            data_generator.plot( "/"+fname, molecules[data_generator.imol], X=Xs[j], Y=Ys[j], entropy=0.0, bXYZ=False, bGroups=False )
            #print "_2"
            #data_generator.plot( "/"+fname, molecules[data_generator.imol], X=None, Y=Ys[j], entropy=0.0, bXYZ=True )

            #print Ys[j]if __name__ == "__main__":

            '''
            fname = "batch_%03i_%03i_" %(i,j)
            data_generator.plot( "/"+fname, molecules[0], Y=Ys[j], entropy=0.0, bPOVray=True, bXYZ=True, bRot=True )
            #subprocess.run("povray Width=800 Height=800 Antialias=On Antialias_Threshold=0.3 Output_Alpha=true %s" %(fname+'.pov') )
            subprocess.call("povray Width=800 Height=800 Antialias=On Antialias_Threshold=0.3 Output_Alpha=true %s" %(fname+'.pov') )
            povname = "./"+molecules[0]+"/"+fname+'.pov'
            cwd = os.getcwd()
            print ">>>>> cwd = os.getcwd() ", cwd
            print ">>>>> povname : ", povname
            os.system( "povray Width=800 Height=800 Antialias=On Antialias_Threshold=0.3 Output_Alpha=true %s" %povname )
            '''
    plt.show()

    '''
    ====== Timing Results ======== ( Machine: GPU:   |  CPU: Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz )
    runTime(Generator_LJC.nextMolecule().1   ) [s]:  0.000192  load atoms
    runTime(Generator_LJC.nextMolecule().2   ) [s]:  0.000264  box,cog
    runTime(Generator_LJC.nextMolecule().3   ) [s]:  0.000633  REAs = getAtomsREA 
    runTime(Generator_LJC.nextMolecule().4   ) [s]:  0.000640  modMolParams  if randomize_parameters 
    runTime(Generator_LJC.nextMolecule().5   ) [s]:  0.000684  cLJs = REA2LJ(REAs) 
    runTime(Generator_LJC.nextMolecule().6   ) [s]:  0.000690  rotJitter 
    runTime(Generator_LJC.nextMolecule().7   ) [s]:  0.000981  pbc, PBCAtoms3D_np 
    runTime(Generator_LJC.nextMolecule().8   ) [s]:  0.002383  forcefield.makeFF                 !!!!!!!!!!!!!!    
    runTime(Generator_LJC.nextMolecule().8-9 ) [s]:  0.001321  projector.prepareBuffers  
    runTime(Generator_LJC.nextMolecule().tot ) [s]:  0.003778  size  [150 150 150   4]
    runTime(Generator_LJC.nextRotation().1   ) [s]:  0.000124  atoms transform(shift,rot)  
    runTime(Generator_LJC.nextRotation().2   ) [s]:  0.000206  top atom 
    runTime(Generator_LJC.nextRotation().3   ) [s]:  0.000229  molCenterAfm  
    runTime(Generator_LJC.nextRotation().4   ) [s]:  0.000267  vtipR0  
    runTime(Generator_LJC.nextRotation().5   ) [s]:  0.002020  scan_pos0s = scanner.setScanRot() 
    runTime(Generator_LJC.nextRotation().6   ) [s]:  0.002110  preHeight 
    runTime(Generator_LJC.nextRotation().7   ) [s]:  0.003439  scanner.run_relaxStrokesTilted()  !!!!!!!!!!!!!!
    runTime(Generator_LJC.nextRotation().8   ) [s]:  0.007660  scanner.run_convolveZ()           !!!!!!!!!!!!!!
    runTime(Generator_LJC.nextRotation().9   ) [s]:  0.010177  X = Fout.z  
    runTime(Generator_LJC.nextRotation().10  ) [s]:  0.011105  poss_ <- scan_pos0s  
    runTime(Generator_LJC.nextRotation().tot ) [s]:  0.013272  size  (128, 128, 20, 4)
    runTime(Generator_LJC.next1().tot        ) [s]:  0.017792
    '''