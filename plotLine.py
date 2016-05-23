#!/usr/bin/python 
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

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


def find_minimum(array,precision=0.0001):
    i=1
    while i<  len(array)-1:
        if (array[i-1] - array[i]) > precision  and ( array[i+1] - array[i]) > precision:
            return i
        i+=1

def selectLine(BIGarray,MIN,MAX,startingPoint, endPoint, nsteps):
    print "Hello world"
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
    print "io", direct
    print "norm", norm_direction
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

    
    
HELP_MSG="""Use this program in the following way:
"""+os.path.basename(main.__file__) +""" -p "XMINxYMINxZMIN" "XMAXxYMAXxZMAX" [-p "XMINxYMINxZMIN" "XMAXxYMAXxZMAX" ...]  """

parser = OptionParser()
parser.add_option( "-k",       action="store", type="float", help="tip stiffenss [N/m]" )
parser.add_option( "--krange", action="store", type="float", help="tip stiffenss range (min,max,n) [N/m]", nargs=3)
parser.add_option( "-q",       action="store", type="float", help="tip charge [e]" )
parser.add_option( "--qrange", action="store", type="float", help="tip charge range (min,max,n) [e]", nargs=3)
parser.add_option( "-a",       action="store", type="float", help="oscilation amplitude [A]" )
parser.add_option( "--arange", action="store", type="float", help="oscilation amplitude range (min,max,n) [A]", nargs=3)
parser.add_option("--image",   action="store", type="float", help="position of the help image (z, xScreen, yScreen)", nargs=3)


parser.add_option("-p", "--points",type=str, help="Point where to perform Z-scan", action="append", nargs=3)
parser.add_option("--disp", type=str, help="print ProbeParticle displacments", action="append", nargs=1)
#parser.add_option( "-y", action="store", type="float", help="format of input file")
#parser.add_option( "--yrange", action="store", type="float", help="y positions of the tip range (min,max,n) [A]", nargs=3)


(options, args) = parser.parse_args()
opt_dict = vars(options)
print options

if options.points==[]:
    sys.exit(HELP_MSG)

print " >> OVEWRITING SETTINGS by params.ini  "
PPU.loadParams( 'params.ini' )
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

print " >> OVEWRITING SETTINGS by command line arguments  "

if opt_dict['krange'] is not None:
	Ks = np.linspace( opt_dict['krange'][0], opt_dict['krange'][1], opt_dict['krange'][2] )
elif opt_dict['k'] is not None:
	Ks = [ opt_dict['k'] ]
else:
	Ks = [ PPU.params['stiffness'][0] ]
# Qs
if opt_dict['qrange'] is not None:
	Qs = np.linspace( opt_dict['qrange'][0], opt_dict['qrange'][1], opt_dict['qrange'][2] )
elif opt_dict['q'] is not None:
	Qs = [ opt_dict['q'] ]
else:
	Qs = [ PPU.params['charge'] ]
# Amps
if opt_dict['arange'] is not None:
	Amps = np.linspace( opt_dict['arange'][0], opt_dict['arange'][1], opt_dict['arange'][2] )
elif opt_dict['a'] is not None:
	Amps = [ opt_dict['a'] ]
else:
	Amps = [ PPU.params['Amplitude'] ]



for iq,Q in enumerate( Qs ):
	for ik,K in enumerate( Ks ):
		dirname = "Q%1.2fK%1.2f" %(Q,K)

                print "Working in {} directory".format(dirname)

                fzs,lvec,nDim,head=GU.loadXSF(dirname+'/OutFz.xsf')
                dfs = PPU.Fz2df( fzs, dz = dz, k0 = PPU.params['kCantilever'], f0=PPU.params['f0Cantilever'], n=Amp/dz )
#                print "TYT", fzs.shape

                for p in options.points:
                    xmin=float(p[0].split('x')[0])
                    ymin=float(p[0].split('x')[1])
                    zmin=float(p[0].split('x')[2])
                    xmax=float(p[1].split('x')[0])
                    ymax=float(p[1].split('x')[1])
                    zmax=float(p[1].split('x')[2])
                    npoints=float(p[2])
                    
                    print opt_dict['disp']
                    if opt_dict['disp'] :
                        print "Displacment {}".format(opt_dict['disp'][0])
                        disp,lvec,nDim,head=GU.loadXSF(dirname+'/PPdisp_'+opt_dict['disp'][0]+'.xsf')
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



                    Fplot=selectLine(BIGarray=fzs, MIN=scan_min,
                               MAX=scan_max,startingPoint=np.array([xmin,ymin,zmin]),
                               endPoint=np.array([xmax,ymax,zmax]),
                               nsteps=npoints)
                    Fplt=np.transpose(Fplot)[1].copy()
                    Lplot=np.transpose(Fplot)[0].copy()
                    F_interp=interp1d(Lplot, Fplt,kind='cubic')
                    fig,ax1 = plt.subplots()
                    ax1.plot(Lplot, Fplt, 'ko', Lplot, F_interp(Lplot), 'k--')
                    ax1.set_xlabel('Coordinate along the selected line ($\AA$)')
                    ax1.set_ylabel('Force (eV/$\AA$)', color='black')
                    for tl in ax1.get_yticklabels():
                        tl.set_color('black')
                    # shifting the df plot 
                        
#                    print "TYT", scan_max[2]
#                    print "TYT", Amp
                    scan_max[2]-=Amp[0]/2.0
                    scan_min[2]+=Amp[0]/2.0
                    DFplot=selectLine(BIGarray=dfs, MIN=scan_min,
                               MAX=scan_max,startingPoint=np.array([xmin,ymin,zmin]),
                               endPoint=np.array([xmax,ymax,zmax]),
                               nsteps=npoints)
                    DFplt=np.transpose(DFplot)[1].copy()
                    Lplot=np.transpose(DFplot)[0].copy()

                    POSplot=np.transpose(DFplot)[2:5].copy()
#                    print POSplot
#                    for k in range(0,dfs.shape[0]-1):
 #                           DFplot[k+(int)(Amp/scan_step[2]/2)]=dfs[-k-1][y_pos][x_pos]
                
                    F_interp=interp1d(Lplot, DFplt,kind='cubic')
                    
                    ax2=ax1.twinx()
#                   min_index= np.argmin(DFplot)
                    min_index= find_minimum(DFplt)
#                    print "MIN", min_index
#                    print DFplot
                    ax2.plot(Lplot, DFplt,'bo', Lplot, F_interp(Lplot), 'b--')
                    axes = plt.gca()
                    ax2.set_ylabel('Frequency shift (Hz)', color='b')
                    for tl in ax2.get_yticklabels():
                        tl.set_color('b')

#                    print Lplot[min_index], DFplt[min_index]
                    ax2.text(Lplot[min_index]+0.02, DFplt[min_index]-1.0, 
                             'x:{:4.2f} ($\AA$); y:{:4.2f} (Hz)'.format(Lplot[min_index], 
                             DFplt[min_index]), style='italic', 
                             bbox={'facecolor':'blue', 'alpha':0.5, 'pad':0})
                    
                    plt.axhline(y=0, color='black', ls='-.')
                    perplane=fig.add_axes([opt_dict['image'][1], opt_dict['image'][2], 0.25, 0.25])
#                    perplane.imshow(dfs[min_index,:, :], origin='image', cmap='gray')

                    zindex=int((opt_dict['image'][0]-scan_min[2]+Amp[0]/2.0)/scan_step[2])
                    perplane.imshow(dfs[zindex,:, :], origin='image', cmap='gray')
                    i=0
                    while i<len(POSplot[0]):
                        perplane.scatter(x=int(POSplot[0][i]/scan_step[0]),
                                         y=int(POSplot[1][i]/scan_step[1]), s=50, c='red', alpha=0.8)
#                        perplane.scatter(x=x_pos, y=y_pos, s=50, c='red', alpha=0.8)
                        x_pos=int(xmin/scan_step[0])
                        y_pos=int(ymin/scan_step[1])
#                        print x_pos, y_pos
#                        print POSplot[0][i], POSplot[1][i]
                        i+=1

                    perplane.axis('off')


                    
                plt.show()
                
