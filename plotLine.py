#!/usr/bin/python 
import os
import sys
import __main__ as main
import numpy as np
import matplotlib.pyplot as plt
#import GridUtils as GU
import pyProbeParticle                as PPU     
import pyProbeParticle.GridUtils      as GU
from scipy.interpolate import interp1d
from optparse import OptionParser
from scipy.interpolate import RegularGridInterpolator
import pyProbeParticle.cpp_utils      as cpp_utils

def selectLine(BIGarray,MIN,MAX,startingPoint, endPoint, nsteps):
    x=np.linspace(MIN[0],MAX[0],BIGarray.shape[2])
    y=np.linspace(MIN[1],MAX[1],BIGarray.shape[1])
    z=np.linspace(MIN[2],MAX[2],BIGarray.shape[0])
    result=[]
    interp = RegularGridInterpolator((z, y, x), BIGarray)
#    print BIGarray.shape
    current_pos=startingPoint
    i=0
    direct=(endPoint-startingPoint)/nsteps
    norm_direction=np.linalg.norm(direct)
    print("io", direct)
    print("norm", norm_direction)
    while i < nsteps :
        current_pos+=direct
#        print current_pos, interp([current_pos[2], current_pos[1],
#                                   current_pos[0]])
        if (current_pos >= MIN).all() and (current_pos <= MAX).all():
            result.append(np.array([norm_direction*i, interp([current_pos[2],
            current_pos[1],current_pos[0]])[0], current_pos[0], current_pos[1],
            current_pos[2]] ))
        i+=1
#    print "TEST", interp([MAX[2], current_pos[1],current_pos[0]])
#    print "TEST", interp([8.0, current_pos[1],current_pos[0]])
    return np.array(result)
    
parser = OptionParser()
parser.add_option("--image",   action="store", type="float", help="position of "
                  "the 2D image (z, xScreen, yScreen)", nargs=3)
parser.add_option("-p", "--points",type=str, help="Point where to perform the "
                  "scan: -p XMINxYMINxZMIN XMAXxYMAXxZMAX", action="append", nargs=3)
parser.add_option("--disp", type=str, help="print ProbeParticle displacments", action="append", nargs=1)
parser.add_option("-f","--data_format" , action="store" , type="string",
                  help="Specify the output format of the vector and scalar "
                  "field. Supported formats are: xsf,npy", default="xsf")
parser.add_option("--nodisp" , action="store_true" ,  help="Do NOT show the "
                  "plots on the screen"     , default=False)


(options, args) = parser.parse_args()
opt_dict = vars(options)
if options.points==[]:
    sys.exit(HELP_MSG)

FFparams=None
if os.path.isfile( 'atomtypes.ini' ):
    print(">> LOADING LOCAL atomtypes.ini")  
    FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 
else:
    import pyProbeParticle.cpp_utils as cpp_utils
    FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
print(" >> OVEWRITING SETTINGS by params.ini  ")
PPU.loadParams( 'params.ini',FFparams=FFparams )
dz  = PPU.params['scanStep'][2]
Amp = [ PPU.params['Amplitude'] ]
scan_min=PPU.params['scanMin']
scan_max=PPU.params['scanMax']
scan_step=PPU.params['scanStep']
gridN=PPU.params['gridN']
gridA=PPU.params['gridA'][0]
gridB=PPU.params['gridB'][1]
gridC=PPU.params['gridC'][2]


MAX=[gridA, gridB, gridC]

K=PPU.params['klat']
Q=PPU.params['charge']
dirname = "Q%1.2fK%1.2f" %(Q,K)

print("Working in {} directory".format(dirname))

fzs,lvec,nDim=GU.load_scal_field(dirname+'/OutFz',data_format=options.data_format)
dfs = PPU.Fz2df( fzs, dz = dz, k0 = PPU.params['kCantilever'], f0=PPU.params['f0Cantilever'], n=Amp/dz )
for p in options.points:
    xmin=float(p[0].split('x')[0])
    ymin=float(p[0].split('x')[1])
    zmin=float(p[0].split('x')[2])
    xmax=float(p[1].split('x')[0])
    ymax=float(p[1].split('x')[1])
    zmax=float(p[1].split('x')[2])
    npoints=float(p[2])
    
    print(opt_dict['disp'])
    if opt_dict['disp'] :
        print("Displacment {}".format(opt_dict['disp'][0]))
        disp_all,lvec,nDim,head=GU.load_vec_field(dirname+'/PPdisp_')
        disp_x,disp_y,disp_z = GU.unpackVecGrid( disp_all ); del disp_all;
        if (opt_dict['disp'][0]=='x'):
            disp = disp_x; del disp_y, disp_z;
        elif (opt_dict['disp'][0]=='y'):
            disp = disp_y; del disp_x, disp_z;
        elif (opt_dict['disp'][0]=='z'):
            disp = disp_z; del disp_x, disp_y;
        DSPplot=selectLine(BIGarray=disp, MIN=scan_min,
                   MAX=scan_max,startingPoint=np.array([xmin,ymin,zmin]),
                   endPoint=np.array([xmax,ymax,zmax]),
                   nsteps=npoints)
        DSPplt=np.transpose(DSPplot)[1].copy()
        Lplot=np.transpose(DSPplot)[0].copy()
        DSP_interp=interp1d(Lplot, DSPplt,kind='cubic')
        plt.plot(Lplot, DSPplt, 'ko',Lplot, DSP_interp(Lplot),'k--')
        plt.axhline(y=0, color='black', ls='-.')
        plt.xlabel('Coordinate along the selected line ($\AA$)')
        plt.ylabel('PP $\Delta$ {} displacement ($\AA$)'.format(opt_dict['disp'][0]), color='black')
        plt.show()


#    print "SCAN MIN,MAX", scan_min,scan_max
    Fplot=selectLine(BIGarray=fzs, MIN=scan_min,
               MAX=scan_max,startingPoint=np.array([xmin,ymin,zmin]),
               endPoint=np.array([xmax,ymax,zmax]),
               nsteps=npoints)
    Fplt=np.transpose(Fplot)[1].copy()
    Lplot=np.transpose(Fplot)[0].copy()
    F_interp=interp1d(Lplot, Fplt,kind='cubic')
    # shifting the df plot 
        
#    print "Amplitude", Amp
    scan_min[2]+=Amp[0]/2.0
    scan_max[2]-=Amp[0]/2.0
    DFplot=selectLine(BIGarray=dfs, MIN=scan_min,
               MAX=scan_max,startingPoint=np.array([xmin,ymin,zmin]),
               endPoint=np.array([xmax,ymax,zmax]),
               nsteps=npoints)
    print(scan_min,scan_max)
    DFplt=np.transpose(DFplot)[1].copy()
    Lplot=np.transpose(DFplot)[0].copy()

    POSplot=np.transpose(DFplot)[2:5].copy()
#                    print POSplot
#                    for k in range(0,dfs.shape[0]-1):
 #                           DFplot[k+(int)(Amp/scan_step[2]/2)]=dfs[-k-1][y_pos][x_pos]

    DF_interp=interp1d(Lplot, DFplt,kind='cubic')
    with open ("x{}-y{}-z{}.dat".format(xmin,ymin,zmin),'w') as f:
        for val in Fplot :
            f.write("{} {} {} {} {} \n".format(val[0],val[1]*1.60217733e3,val[2],val[3],val[4]))
    
    if not opt_dict['nodisp'] :
        fig,ax1 = plt.subplots()
        ax1.plot(Lplot, Fplt*1.60217733e3, 'ko', Lplot,
        F_interp(Lplot)*1.60217733e3, 'k--')
        ax1.set_xlabel('Coordinate along the selected line ($\AA$)')
        ax1.set_ylabel('Force (eV/$\AA$)', color='black')
        for tl in ax1.get_yticklabels():
            tl.set_color('black')
        ax2=ax1.twinx()
        print(DFplot)
        ax2.plot(Lplot, DFplt,'bo', Lplot, DF_interp(Lplot), 'b--')
        axes = plt.gca()
        ax2.set_ylabel('Frequency shift (Hz)', color='b')
        for tl in ax2.get_yticklabels():
            tl.set_color('b')
        plt.axhline(y=0, color='black', ls='-.')
        perplane=fig.add_axes([opt_dict['image'][1], opt_dict['image'][2], 0.25, 0.25])
        zindex=int((opt_dict['image'][0]-scan_min[2]+Amp[0]/2.0)/scan_step[2])
        perplane.imshow(dfs[zindex,:, :], origin='image', cmap='gray')
        i=0
        while i<len(POSplot[0]):
            perplane.scatter(x=int((POSplot[0][i]-scan_min[0])/scan_step[0]),
                             y=int((POSplot[1][i]-scan_min[1])/scan_step[1]), s=50, c='red', alpha=0.8)
            x_pos=int(xmin/scan_step[0])
            y_pos=int(ymin/scan_step[1])
            i+=1
        perplane.axis('off')
        plt.show()
