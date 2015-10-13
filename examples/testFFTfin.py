#import scipy.ndimage.interpolation as itp
import matplotlib.pyplot as plt
import numpy as np
from xsfutil import *
from STHM_Utils import *
from libFFTfin import *

Fx, lvec, nDim, head = loadXSF('LOCPOT.xsf')

sampleSize = getSampleDimensions(lvec)
dims = (nDim[2], nDim[1], nDim[0])

xsize, dx = getSize('x', dims, sampleSize)
ysize, dy = getSize('y', dims, sampleSize)
zsize, dz = getSize('z', dims, sampleSize)

print 'dx, dy, dz = ', dx, dy, dz
print 'xsize, ysize, zsize = ', xsize, ysize, zsize

dd = (dx, dy, dz)

extent = ( -xsize/2, xsize/2,  -ysize/2, ysize/2 )
ilist = range(50,70,2) 

#FxA = itp.rotate(Fx, 30, axes=(1, 2))
#
#plotWithAtoms( FxA, ilist, extent, dz = dz, cmap = 'jet', withcolorbar=True )

def testInterA(data, dd, coordX, coordY, coordZ, sizes):
    "get coordinates (lenghts) and returns approximate values in these coord."    
    
    dx, dy, dz = dd
    
    k1 = np.floor(coordX/dx) % sizes[0]    
    k2 = np.floor(coordY/dy) % sizes[1]    
    k3 = np.floor(coordZ/dz) % sizes[2]
    
#    value = data[k1, k2, k3]
    value = data[k3, k2, k1]
    return value    
  
def interpolateAlt(data, dd, coordX, coordY, coordZ, dims):
    'linear interpolation of data in coordinates coord based on nearby indices'    
    
    dx, dy, dz = dd
    xdim, ydim, zdim = dims

    # !!! coordinates coordX, etc. are lengths !!!

    coordXind = (coordX/dx) % xdim
    coordYind = (coordY/dy) % ydim
    coordZind = (coordZ/dz) % zdim

    k1d = np.floor(coordXind)
    k2d = np.floor(coordYind)
    k3d = np.floor(coordZind)

#    k1d = np.round(coordXind) % xsize
#    k2d = np.round(coordYind) % ysize
#    k3d = np.round(coordZind) % zsize

    k1u = (k1d + 1) % xdim
    k2u = (k2d + 1) % ydim
    k3u = (k3d + 1) % zdim
    
    p000 = data[k3d, k2d, k1d] # x <-> z ???
    p001 = data[k3d, k2d, k1u] # x <-> z ???
    p010 = data[k3d, k2u, k1d] # x <-> z ???
    p100 = data[k3u, k2d, k1d] # x <-> z ???
    
    p011 = data[k3d, k2u, k1u] # x <-> z ???
    p101 = data[k3u, k2d, k1u] # x <-> z ???
    p110 = data[k3u, k2u, k1d] # x <-> z ???
    p111 = data[k3u, k2u, k1u] # x <-> z ???
   
    app1_00 = p000 + (coordXind - k1d)*(p100 - p000)
    app1_10 = p010 + (coordXind - k1d)*(p110 - p010)
    app1_01 = p001 + (coordXind - k1d)*(p101 - p001)
    app1_11 = p011 + (coordXind - k1d)*(p111 - p011)

    app2_0 = app1_00 + (coordYind - k2d)*(app1_10 - app1_00)
    app2_1 = app1_01 + (coordYind - k2d)*(app1_11 - app1_01)
    
    app3 = app2_0 + (coordZind - k3d)*(app2_1 - app2_0)
    
    return app3
  
  
def getCutXAlt(data, yzero, zzero, sampleSize, dd, num, delta, dims):
    
    # in order to compare different samples delta cannot depend on sampleSize !!!

    mat = getNormalizedBasisMatrix(sampleSize).getI().getT()
    cut = zeros((2, num))

    coord_y = yzero
    coord_z = zzero

    def getSkewCoord(coord_x, coord_y, coord_z, mat):
        skewCoords = [0, 0, 0]
        skewCoords[0] = mat[0,0]*coord_x + mat[0,1]*coord_y + mat[0,2]*coord_z
        skewCoords[1] = mat[1,0]*coord_x + mat[1,1]*coord_y + mat[1,2]*coord_z
        skewCoords[2] = mat[2,0]*coord_x + mat[2,1]*coord_y + mat[2,2]*coord_z
        return skewCoords
    
    def getValue(data, skewCoords, dd, dims):
#        value = interpolateAlt(data, dd, skewCoords[0], skewCoords[1], skewCoords[2], dims)
        value = testInterA(data, dd, skewCoords[0], skewCoords[1], skewCoords[2], dims)
        return value

    for i in range(num):
        coord_x = i*delta
        skewCoords = getSkewCoord(coord_x, coord_y, coord_z, mat)
        value = getValue(data, skewCoords, dd, dims)
        cut[0, i] = coord_x
        cut[1, i] = value
        
    return cut
 
  # BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!
  # BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!
  # BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!
  # BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!
  # BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!
    
yzero = 0.0
zzero = 0.0
num_x_inp = 150
delta_x_inp = 0.07

data = np.array(Fx)

cut = getCutXAlt(data, yzero, zzero, sampleSize, dd, num_x_inp, delta_x_inp, dims)

print 'cut[1].max(), cut[1].min() = ', cut[1].max(), cut[1].min()

plt.plot(cut[0], cut[1], '.-')

# BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!
# BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!
# BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!
# BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!
# BACHA NA ZAMENU x-OVE A z-OVE SOURADNICE !!!

plt.show()
