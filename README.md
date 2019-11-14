
# Probe Particle Model

This is implementation of efficient and simple model for simulation of High-resolution atomic force microscopy (AFM), scanning probe microscopy (STM) and inelastic tunneling microscopy (IETS) images using classical forcefileds.

### References
* [Prokop Hapala, Georgy Kichin, Christian Wagner, F. Stefan Tautz, Ruslan Temirov, and Pavel Jelínek, Mechanism of high-resolution STM/AFM imaging with functionalized tips, Phys. Rev. B 90, 085421 – Published 19 August 2014](http://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.085421)
* [Prokop Hapala, Ruslan Temirov, F. Stefan Tautz, and Pavel Jelínek, Origin of High-Resolution IETS-STM Images of Organic Molecules with Functionalized Tips, Phys. Rev. Lett. 113, 226101 – Published 25 November 2014,](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.226101) 


## Interactive Real-time GUI using OpenGL

While C++ core can computed typical 3D stack of ~40 images in ~1 minute, using power of modern GPUs additional acceleration by factor of ~100x can be achieved. This makes it feasible to use PPmodel in form of an interactive GUI where simulated images are immediately updated upon change of experimental parameters (e.g. tip charge and striffness) or of input atomic geometry (e.g. positions and atomic charges). This may be very usefull for experimentalist which just want quick idea how an AFM picture they youst measure correspond to the atomistic model they consider.

The OpenGL GUI version si more-or-less finished with most of functionality implemented. The code is however not yet merged to master branch. It can be found in independent branch here: 
https://github.com/ProkopHapala/ProbeParticleModel/tree/OpenCL

#### Usage:

 * go to `ProbeParticleModel/tree/OpenCL/tests/testFieldOCL`
 * run `> python GUI.py`
 
 Following interface should appear:
 ![GUI interface example](doc/OpenCL/GUI.jpg?raw=true "")
  
* When you change any number in input box the image will update. 
* The initial images is forcefield. Only after update of some relaxation related parameter (e.g. `K`,`Q`) relaxed AFM image (i.e. `df`) is rendered.
* each update means recalculation of the whole 3D volume Force-field and relaxed `df`. So it may take up to several seconds depending on your GPU (on my *nVidia GT 960M* it takes `~0.1s` )
* Amplitude for conversion `Fz -> df` is determined by `nAmp` parameter, which is the number of slices used in giesible formula. By default the grid spacing is `0.1A` (`10 pm`) therefore `nAmp=15` means peak-to-peak amplitude `1.5A` (`150pm`).

* You can also interactively change interactively the atomic geometry and charges of sample. To do so click `Edit` button. Following interface window should open:
 ![GUI interface example](doc/OpenCL/edit.jpg?raw=true "")
 
after you finish your changes in the geometry click `Update` button.

#### Prerequsities:
 * python 2.7 with packages: 
    * numpy
    * PyOpenCL 
 * GPU with OpenCL support and Correctly instaled GPU drivers with OpenCL support



  
