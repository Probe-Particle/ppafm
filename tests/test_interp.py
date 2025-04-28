import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import pyProbeParticle.interp as interp

'''
xs = np.linspace(0.0,1.2,100)
ys = interp.sample_kernel_func(xs, kind=2)
plt.plot(xs,ys)
plt.grid()
plt.ylim(0.0,2.0)
plt.show()
exit(0)
'''


data=np.array([
    [1.0, 1.0,    5.0],
    [2.0, 1.0,   -5.0],
    [1.0, 2.0,   8.0],
    [2.0, 2.0,   12.0],
])

data_points = data[:,0:2].copy()
data_vals   = data[:,2].copy()

nx,ny = 50,50
xs = np.linspace(0.0,3.0,nx)
ys = np.linspace(0.0,3.0,ny)
Xs,Ys = np.meshgrid( xs, ys )
ps = np.zeros((len(xs)*len(ys),2))
ps[:,0] = Xs.flatten()
ps[:,1] = Ys.flatten()


nNeighMax = 8
out_vals, out_neighs, out_weights = interp.interpolate_2d( data_points, data_vals, ps, Rcut=2.0, nNeighMax=nNeighMax, bBasis=True, mode=4 )

out_weights = out_weights.reshape((nx,ny,nNeighMax))

nrow = int(np.sqrt(nNeighMax)); ncol = int(np.ceil(nNeighMax/nrow)) 

sh = out_neighs.shape
extent = (xs[0], xs[-1], ys[0], ys[-1] )

plt.figure()
plt.imshow( out_vals.reshape((nx,ny)), origin='lower', extent=extent )
plt.plot( data_points[:,0], data_points[:,1], '.' )
plt.colorbar()


plt.figure()
wsum = np.sum(out_weights, axis=2)
wmax = np.max(out_weights, axis=2)
wratio = wmax/wsum
plt.imshow( wratio, origin='lower', extent=extent )
plt.plot( data_points[:,0], data_points[:,1], '.' )
plt.colorbar()

 


# fig,axs = plt.subplots( nrow, ncol, figsize=(3*ncol,3*nrow) )
# axs_ = axs.flat
# for i in range(sh[-1]):
#     wis = out_weights[:,:,i]
#     axs_[i].imshow( wis, origin='lower', extent=extent )
#     axs_[i].plot( data_points[:,0], data_points[:,1], '.' )
#     axs_[i].set_title( "w_%i" %i )


plt.show()


   



