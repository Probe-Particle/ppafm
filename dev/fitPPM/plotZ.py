#!/usr/bin/python
# This is a sead of simple plotting script which should get AFM frequency delta 'df.xsf' and generate 2D plots for different 'z'

import os
import sys
from optparse import OptionParser

import __main__ as main
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import ppafm as PPU
from ppafm import io


def find_minimum(array,precision=0.0001):
    i=1
    while i<  len(array):
        if (array[i-1] - array[i]) > precision  and ( array[i+1] - array[i]) > precision:
            return i
        i+=1



HELP_MSG="""Use this program in the following way:
"""+os.path.basename(main.__file__) +""" -p "X1xY1" [-p "X2xY2" ...]  """

parser = OptionParser()
parser.add_option( "-k",       action="store", type="float", help="tip stiffenss [N/m]" )
parser.add_option( "--krange", action="store", type="float", help="tip stiffenss range (min,max,n) [N/m]", nargs=3)
parser.add_option( "-q",       action="store", type="float", help="tip charge [e]" )
parser.add_option( "--qrange", action="store", type="float", help="tip charge range (min,max,n) [e]", nargs=3)
parser.add_option( "-a",       action="store", type="float", help="oscilation amplitude [A]" )
parser.add_option( "--arange", action="store", type="float", help="oscilation amplitude range (min,max,n) [A]", nargs=3)



parser.add_option("-p", "--points", default=[], type=str, help="Point where to perform Z-scan", action="append")
parser.add_option( "--npy" , action="store_true" ,  help="load and save fields in npy instead of xsf"     , default=False)

#parser.add_option( "-y", action="store", type="float", help="format of input file")
#parser.add_option( "--yrange", action="store", type="float", help="y positions of the tip range (min,max,n) [A]", nargs=3)


(options, args) = parser.parse_args()
opt_dict = vars(options)
print(options)
if options.npy:
    format ="npy"
else:
    format ="xsf"

if options.points==[]:
    sys.exit(HELP_MSG)

print(" >> OVEWRITING SETTINGS by params.ini  ")
PPU.loadParams( 'params.ini' )
dz  = PPU.params['scanStep'][2]
Amp = [ PPU.params['Amplitude'] ]
scan_max=PPU.params['scanMax'][2]
scan_min=PPU.params['scanMin'][2]
scan_step=PPU.params['scanStep'][2]

print(" >> OVEWRITING SETTINGS by command line arguments  ")

if opt_dict['krange'] is not None:
	Ks = np.linspace( opt_dict['krange'][0], opt_dict['krange'][1], opt_dict['krange'][2] )
elif opt_dict['k'] is not None:
	Ks = [ opt_dict['k'] ]
else:
	Ks = [ PPU.params['klat'] ]
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

                print("Working in {} directory".format(dirname))

                fzs,lvec,nDim,head=io.load_scal_field(dirname+'/OutFz', format=format)
                dfs = PPU.Fz2df( fzs, dz = dz, k0 = PPU.params['kCantilever'], f0=PPU.params['f0Cantilever'], n=Amp/dz )
                for p in options.points:
                    x=float(p.split('x')[0])
                    y=float(p.split('x')[1])
                    x_pos=int(x/scan_step)
                    y_pos=int(y/scan_step)

                    Zplot=np.zeros(fzs.shape[0])
                    Fplot=np.zeros(fzs.shape[0])
                    DFplot=np.zeros(fzs.shape[0])


                    for k in range(0,fzs.shape[0]):
                            Fplot[k]=fzs[-k-1][y_pos][x_pos]
                            Zplot[k]=scan_max-scan_step*k


                    # shifting the df plot
                    for k in range(0,dfs.shape[0]-1):
                            DFplot[k+(int)(Amp/scan_step/2)]=dfs[-k-1][y_pos][x_pos]


                    xnew = np.linspace(Zplot[0], Zplot[-1], num=41, endpoint=True)
                    F_interp=interp1d(Zplot, Fplot,kind='cubic')
                    fig,ax1 = plt.subplots()
                    ax1.plot(Zplot, Fplot, 'ko', xnew, F_interp(xnew), 'k--')
                    ax1.set_xlabel(r'Z coordinate of the tip ($\AA$)')
                    ax1.set_ylabel(r'Force (eV/$\AA$)', color='black')
                    for tl in ax1.get_yticklabels():
                        tl.set_color('black')




                    F_interp=interp1d(Zplot, DFplot,kind='cubic')

                    ax2=ax1.twinx()
#                   min_index= np.argmin(DFplot)
                    min_index= find_minimum(DFplot)
#                    print "MIN", min_index
#                    print DFplot
                    ax2.plot(Zplot, DFplot,'bo', xnew, F_interp(xnew), 'b--')
                    axes = plt.gca()
                    ax2.set_ylabel('Frequency shift (Hz)', color='b')
                    for tl in ax2.get_yticklabels():
                        tl.set_color('b')
                    ax2.text(Zplot[min_index]+0.02, DFplot[min_index]-1.0,
                             r'x:{:4.2f} ($\AA$); y:{:4.2f} (Hz)'.format(Zplot[min_index],
                             DFplot[min_index]), style='italic',
                             bbox={'facecolor':'blue', 'alpha':0.5, 'pad':0})

                    plt.axhline(y=0, color='black', ls='-.')
                    perplane=fig.add_axes([0.65, 0.6, 0.25, 0.25])
#                    perplane.imshow(dfs[min_index+int(0.5/scan_step),:, :], origin='upper', cmap='gray')
#                    perplane.imshow(dfs[len(Zplot)-min_index-(int)(Amp/scan_step/2)-5, :, :], origin='upper', cmap='gray')
                    perplane.imshow(dfs[len(Zplot)-min_index-(int)(Amp/scan_step/2)-int(1.0/scan_step), :, :], origin='upper', cmap='gray')

                    perplane.scatter(x=x_pos, y=y_pos, s=50, c='red', alpha=0.8)
                    perplane.axis('off')



                plt.show()
