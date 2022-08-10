
# Probe Particle Model

This is implementation of efficient and simple model for simulation of High-resolution atomic force microscopy (AFM), scanning probe microscopy (STM) and inelastic tunneling microscopy (IETS) images using classical forcefileds.

There are two versions of the code - 

* currently developed Python/C++ version in PyProbe_nonOrtho  (branch *master* ); 
  * to get quick esence of this model you can also try web interface hostet here: http://nanosurf.fzu.cz/ppr/
  * for more details see wikipage: https://github.com/ProkopHapala/ProbeParticleModel/wiki
* Legacy fortran version in SHTM_springTip2 (branch *fortran* ); 
  * more detailed description o the fortran version is here: http://nanosurf.fzu.cz/wiki/doku.php?id=probe_particle_model 
  
## WIKI

More details about the code and its usage can be found [here](https://github.com/ProkopHapala/ProbeParticleModel/wiki)
  
### References
* [Prokop Hapala, Georgy Kichin, Christian Wagner, F. Stefan Tautz, Ruslan Temirov, and Pavel Jelínek, Mechanism of high-resolution STM/AFM imaging with functionalized tips, Phys. Rev. B 90, 085421 – Published 19 August 2014](http://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.085421)
* [Prokop Hapala, Ruslan Temirov, F. Stefan Tautz, and Pavel Jelínek, Origin of High-Resolution IETS-STM Images of Organic Molecules with Functionalized Tips, Phys. Rev. Lett. 113, 226101 – Published 25 November 2014,](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.226101) 

## New Interactive real-time GUI using OpenCL

While C++ core can compute typical 3D stack of ~40 images in ~1 minute, using power of modern GPUs additional acceleration by factor of ~100x can be achieved. This makes it feasible to use PPmodel in form of an interactive GUI where simulated images are immediately updated upon change of experimental parameters (e.g. tip charge and stiffness) or of input atomic geometry (e.g. positions and atomic charges). This may be very useful for experimentalist who want to quickly test how AFM images they measure correspond to the atomistic model they consider.

The OpenCL GUI version is more-or-less finished with most of functionality implemented. The code is however not yet merged to master branch. It can be found in independent branch here: 
https://github.com/ProkopHapala/ProbeParticleModel/tree/OpenCL_py3
