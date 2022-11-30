#!/usr/bin/python
import sys
import numpy as np
import os
import __main__ as main

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
import ppafm                as PPU     
from   ppafm            import basUtils
from   ppafm            import elements   
import ppafm.GridUtils      as GU
#import ppafm.core          as PPC
import ppafm.HighLevel      as PPH
import ppafm.fieldFFT       as fFFT

HELP_MSG="""Use this program in the following way:
%s -i <filename> 

Supported file fromats are:
   * xyz 
""" %os.path.basename(main.__file__)


from optparse import OptionParser

parser = OptionParser()
parser.add_option( "--noProbab", action="store_false",  help="probability False", default=True )
parser.add_option( "--noForces", action="store_false",  help="Forces False"     , default=True )
parser.add_option( "--current" , action="store_true" ,  help="current True"     , default=False)
parser.add_option("-f","--data_format" , action="store" , type="string", help="Specify the output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
(options, args) = parser.parse_args()
opt_dict = vars(options)

kBoltz = 8.617332478e-5   # [ eV / K ]

# ============= functions

def getXYZ( nDim, cell ):
    '''
    getXYZ( nDim, cell ):
    X,Y,Z - output: three dimensional arrays with x, y, z coordinates as value
    '''
    dcell = np.array( [ cell[0]/nDim[2], cell[1]/nDim[1], cell[2]/nDim[0] ]  )
    print(" dcell ", dcell) 
    CBA = np.mgrid[0:nDim[0],0:nDim[1],0:nDim[2]].astype(float) # grid going: CBA[z,x,y]
    X = CBA[2]*dcell[0, 0] + CBA[1]*dcell[1, 0] + CBA[0]*dcell[2, 0]
    Y = CBA[2]*dcell[0, 1] + CBA[1]*dcell[1, 1] + CBA[0]*dcell[2, 1]
    Z = CBA[2]*dcell[0, 2] + CBA[1]*dcell[1, 2] + CBA[0]*dcell[2, 2]
    return X,Y,Z

def getProbeDensity( pos,  X, Y, Z, sigma ):
    '''
    getProbeDensity( pos,  X, Y, Z, sigma ):
    pos - position of the tip
    X,Y,Z - input: three dimensional arrays with x, y, z coordinates as value
    sigma - FWHM of the Gaussian function
    '''
    r2      = (X-pos[0])**2 + (Y-pos[1])**2 + (Z-pos[2])**2
    radial  = np.exp( -r2/( sigma**2 ) )
    return radial
    #return (X-pos[0])**2

def getProbeTunelling( pos,  X, Y, Z, beta=1.0 ):
    '''
    getProbeTunelling( pos,  X, Y, Z, sigma ):
    pos - position of the tip
    X,Y,Z - input: three dimensional arrays with x, y, z coordinates as value
    beta - decay in eV/Angstrom
    '''
    r2      = (X-pos[0])**2 + (Y-pos[1])**2 + (Z-pos[2])**2
    radial  = np.exp( -beta * np.sqrt(r2) )
    return radial

def limitE( E, E_cutoff ):
    '''
    limitE( E, E_cutoff ):
    exclude too high or infinite energies
    '''
    Emin    = E.min()
    E      -= Emin
    mask    = ( E > E_cutoff )
    E[mask] = E_cutoff

def W_cut(W,nz=100,side='up',sm=10):
    '''
    W_cut(W,nz=100,side='up',sm=10):
    W - incomming potential
    nz - z segment, where is the turning point of the fermi function
    'up' cuts up than nz
    'down' cuts down than nz
    sm - smearing = width of the fermi function in number of z segments
    '''
    ndim=W.shape
    print(ndim)
    if (side=='up'):
        for iz in range(ndim[0]):
            #print iz, 1/(np.exp((-iz+nz)*1.0/sm) + 1)
            W[iz,:,:] *= 1/(np.exp((-iz+nz)*1.0/sm) + 1)
    if (side=='down'):
        for iz in range(ndim[0]):
            #print iz, 1/(np.exp((iz-nz)*1.0/sm) + 1)
            W[iz,:,:] *= 1/(np.exp((iz-nz)*1.0/sm) + 1)
    return W;

# ============== setup 

T = 10.0 # [K]

beta = 1/(kBoltz*T)   # [eV]
print("T= ", T, " [K] => beta ", beta/1000.0, "[meV] ") 

#E_cutoff = 32.0 * beta
E_cutoff = 18.0 * beta

wGauss =  2.0
Egauss = -0.01


# =============== main

if options.noProbab :
    print(" ==== calculating probabilties ====")
    # --- tip
    V_tip,   lvec, nDim = GU.load_scal_field('tip/VLJ',data_format=options.data_format)
    #cell   = np.array( [ lvec[1],lvec[2],lvec[3] ] ); print "nDim ", nDim, "\ncell ", cell
    #X,Y,Z  = getXYZ( nDim, cell )
    #V_tip = V_tip*0 + Egauss * getProbeDensity( (cell[0,0]/2.+cell[1,0]/2.,cell[1,1]/2,cell[2,2]/2.-3.8), X, Y, Z, wGauss ) # works for tip (the last flexible tip apex atom) in the middle of the cell
    limitE( V_tip,  E_cutoff ) 
    W_tip  = np.exp( -beta * V_tip  )
    #W_tip = W_cut(W_tip,nz=95,side='down',sm=5)
    del V_tip;
    GU.save_scal_field ( 'W_tip',  W_tip,    lvec, data_format=options.data_format)

    # --- sample
    V_surf,  lvec, nDim = GU.load_scal_field('sample/VLJ',data_format=options.data_format)
    limitE( V_surf, E_cutoff ) 
    W_surf = np.exp( -beta * V_surf )
    #W_surf=W_cut(W_surf,nz=50,side='up',sm=1)
    del V_surf; 
    GU.save_scal_field ( 'W_surf', W_surf,   lvec, data_format=options.data_format)

#=================== Force

if options.noForces :
    print(" ==== calculating Forces ====") 
    if (options.noProbab==False) :
        print(" ==== loading probabilties ====") 
        # --- tip
        W_tip,   lvec, nDim = GU.load_scal_field('W_tip',data_format=options.data_format)
        # --- sample
        W_surf,  lvec, nDim = GU.load_scal_field('W_surf',data_format=options.data_format)

    W_tip = np.roll(np.roll(np.roll(W_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell
    FF_tmp, lvec, nDim = GU.load_vec_field('tip/FFLJ',data_format=options.data_format)
    Fx_tip, Fy_tip, Fz_tip = GU.unpackVecGrid( FF_tmp )
    del FF_tmp;

    # Fz:
    Fz_tip = np.roll(np.roll(np.roll(Fz_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell
    F1=fFFT.Average_tip( Fz_tip , W_surf, W_tip  )
    #GU.saveXSF        ( 'FFboltz_z.xsf', F1, lvec)#, echo=True )

    # Fx:
    Fx_tip = np.roll(np.roll(np.roll(Fx_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell
    F2=fFFT.Average_tip( Fx_tip , W_surf, W_tip  )
    #GU.saveXSF        ( 'FFboltz_x.xsf', F1, lvec)#, echo=True )

    # Fy:
    Fy_tip = np.roll(np.roll(np.roll(Fy_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell
    F3=fFFT.Average_tip( Fy_tip , W_surf, W_tip  )
    #GU.saveXSF        ( 'FFboltz_y.xsf', F1, lvec)#, echo=True )
    FF_boltz = GU.packVecGrid(F3,F2,F1)
    GU.save_vec_field('FFboltz',FF_boltz,lvec, data_format=options.data_format)
    del F1; del F2; del F3; del FF_boltz; del Fz_tip; del Fy_tip; del Fx_tip;


    print("x,y & z forces for the Boltzmann distribution of moving particle stored")

'''
# surface just for debugging
#Fz_surf, lvec, nDim, head = GU.loadXSF('sample/FFLJ_z.xsf')
#F2=Average_surf( Fz_surf, W_surf, W_tip )
#GU.saveXSF        ( 'Fz_surf.xsf', F2, lvec, echo=True )
'''
#=================== Current


if options.current :
    if ((options.noProbab==False)and(options.noForces==False)) :
        print(" ==== loading probabilties ====") 
        # --- tip
        W_tip,   lvec, nDim = GU.load_scal_field('W_tip',data_format=options.data_format)
        W_tip = np.roll(np.roll(np.roll(W_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell
        # --- sample
        W_surf,  lvec, nDim = GU.load_scal_field('W_surf',data_format=options.data_format)

    if ((options.noProbab)and(options.noForces==False)) :
        W_tip = np.roll(np.roll(np.roll(W_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2) # works for tip (the last flexible tip apex atom) in the middle of the cell

    print(" ==== calculating current ====") 
    cell   = np.array( [ lvec[1],lvec[2],lvec[3] ] ); print("nDim ", nDim, "\ncell ", cell)
    X,Y,Z  = getXYZ( nDim, cell )
    T_tip = getProbeTunelling( (cell[0,0]/2.+cell[1,0]/2.,cell[1,1]/2,cell[2,2]/2.) ,  X, Y, Z, beta=1.14557 )  #beta decay in eV/Angstom for WF = 5.0 eV;  works for tip (the last flexible tip apex atom) in the middle of the cell
    T_tip = np.roll(np.roll(np.roll(T_tip,nDim[0]/2, axis=0),nDim[1]/2, axis=1),nDim[2]/2, axis=2)
    T=fFFT.Average_tip( (-1)*T_tip, W_surf, W_tip )                  # T stands for hoppings
    del T_tip;
    print(T.shape)
    print((T**2).shape)
    GU.save_scal_field ( 'I_boltzmann', T**2,   lvec, data_format=options.data_format) # I ~ T**2 

print(" ***** ALL DONE ***** ")
