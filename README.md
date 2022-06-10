
# Probe Particle Model

This is implementation of efficient and simple model for simulation of High-resolution atomic force microscopy (AFM), scanning probe microscopy (STM) and inelastic tunneling microscopy (IETS) images using classical forcefileds.

### References
* [Prokop Hapala, Georgy Kichin, Christian Wagner, F. Stefan Tautz, Ruslan Temirov, and Pavel Jelínek, Mechanism of high-resolution STM/AFM imaging with functionalized tips, Phys. Rev. B 90, 085421 – Published 19 August 2014](http://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.085421)
* [Prokop Hapala, Ruslan Temirov, F. Stefan Tautz, and Pavel Jelínek, Origin of High-Resolution IETS-STM Images of Organic Molecules with Functionalized Tips, Phys. Rev. Lett. 113, 226101 – Published 25 November 2014,](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.226101) 


## Interactive Real-time GUI using OpenGL

While C++ core can computed typical 3D stack of ~40 images in ~1 minute, using power of modern GPUs additional acceleration by factor of ~100x can be achieved. This makes it feasible to use PPmodel in form of an interactive GUI where simulated images are immediately updated upon change of experimental parameters (e.g. tip charge and striffness) or of input atomic geometry (e.g. positions and atomic charges). This may be very usefull for experimentalist which just want quick idea how an AFM picture they youst measure correspond to the atomistic model they consider.

The OpenGL GUI version si more-or-less finished with most of functionality implemented. The code is however not yet merged to master branch. It can be found in independent branch here: 
https://github.com/ProkopHapala/ProbeParticleModel/tree/OpenCL

#### Installation:

Install prerequisites (Ubuntu):
```sh
sudo apt install git python3-pip python3-pyqt5
pip install matplotlib numpy pyopencl reikna
```

Additionally an OpenCL Installable Client Driver (ICD) for your compute device is required:
* Nvidia GPU: comes with the standard Nvidia driver (nvidia-driver-xxx)
* AMD GPU: `sudo apt install mesa-opencl-icd`
* Intel HD Graphics: `sudo apt install intel-opencl-icd`
* CPU: `sudo apt install pocl-opencl-icd`

Clone the repository and navigate to the cloned directory
```sh
git clone https://github.com/ProkopHapala/ProbeParticleModel.git -b gui
cd ProbeParticleModel
```

Run the GUI application:
```sh
./GUILJ.py
```
  
#### Usage:
* Open a file by clicking `Open File...` at the bottom or provide an input file as a command line argument using the `-i` or `--input` option. The input file can be .xyz geometry file (possibly with point charges) or a .xsf or .cube Hartree potential file. Loading large files may take some time.
* Changing any number in any input box will automatically update the image. There are also presets for some commonly used tip configurations.
* Hover mouse cursor over any parameter for a tooltip explaining the meaning of the parameter.
* Save the current image or df data by clicking the `Save Image...` or `Save df...` buttons at the bottom.
* In case there are multiple OpenCL devices installed on the system, use the `-l` or `--list-devices` option to list available devices and choose the device using the `-d` or `--device` option with the device platform number as the argument.

