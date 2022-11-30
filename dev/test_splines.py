#!/usr/bin/python -u

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.split(sys.path[0])[0]) #;print(sys.path[-1])
#import ppafm                as PPU 
import ppafm.core           as PPC



# =========== uniform spline

ys  = np.array   ( [  0.0, 0.2, -0.3,  0.0 ] )
dys = np.array   ( [  0.8, 0.8, -1.2, -1.2 ] )
xs_ = np.linspace(   -0.89, 0.59, 100        )
ydys = np.transpose( np.array([ys, dys]) ).copy()
#print "ydys" = ydys

x0  =-0.9; dx = 0.5;
ys_ = PPC.subsample_uniform_spline( x0, dx, ydys, xs_ )
xs  = np.array( list(range(len(ys))) )*dx + x0

plt.figure()
plt.plot( xs,  ys ,        'o'   );
plt.plot( xs,  ys+0.1*dys , '.'  );
plt.plot( xs_, ys_, '-' );
plt.grid()

# =========== non-uniform spline

xs  = np.array   ( [ -0.9, -0.6,  0.4,  0.6 ] )
#ys  = np.array   ( [  0.0, 0.2, -0.3,  0.0 ] )
#dys = np.array  ( [  0.8, 0.8, -1.2, -1.2 ] )
#dys = np.array   ( [  0.0, 0.0, 0.0, 0.0   ] )
#xs_ = np.linspace(   -0.89, 0.59, 20           )

ys_ = PPC.subsample_nonuniform_spline( xs, ydys, xs_ )

plt.figure()
plt.plot( xs,  ys ,        'o'   );
plt.plot( xs,  ys+0.1*dys , '.'  );
plt.plot( xs_, ys_, '-'          );
plt.grid()

#print "xs_", xs_
#print "ys_", ys_

plt.show()



