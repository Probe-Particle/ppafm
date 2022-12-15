
# Probe Particle Model (PPM)

Simple and efficient **simulation software for high-resolution atomic force microscopy** (**HR-AFM**) and other scanning probe microscopy (SPM) techniques with sub-molecular resolution (STM, IETS, TERS). It simulates deflection of the particle attached to the tip (typically CO molecule, but also e.g. Xe, Cl-, H2O and others).

### Further information
- Publications: https://github.com/Probe-Particle/ProbeParticleModel#notable-publications-using-probe-particle-model
- Wiki: https://github.com/Probe-Particle/ProbeParticleModel/wiki
- API documentation: https://ppafm.readthedocs.io/en/latest/

## Flavors of PPM

Since 2014 PPM developed into the toolbox of various methodologies adjusted for a particular use case. 

1. **CPU version:** - Original implementation using Python & C/C++. It can simulate a typical AFM experiment (3D stack of AFM images) in ~1 minute. It is the base version for the development of new features and methodology. All available simulation models are implemented in this version, including:
   1. **Point charge electrostatics + Lennard-Jones:** Original fully classical implementation allows the user to set up calculation without any ab-initio input simply by specifying atomic positions, types and charges.
   2. **Hartree-potential electrostatics + Lennard-Jones:** Electrostatics is considerably improved by using Hartree potential from DFT calculation (e.g. LOCPOT from VASP) and using the Quadrupole model for CO-tip. We found this crucial to properly simulate polar molecules (e.g. H2O clusters, carboxylic acids, PTCDA) which exhibit strong electrostatic distortions of AFM images. Thanks to implementation using fast Fourier transform (FFT) this improvement does not increase the computational time (still ~1 minute), as long as the input electrostatic field is accessible.
   3. **Hartree-potential electrostatics + Density overlap:** Further accuracy improvement is achieved when Pauli repulsion between electron shells of atoms is modeled by the overlap between electron density of tip and sample. This repulsive term replaces the repulsive part of Lennard-Jones while the attractive part (C6) remains. This modification considerably improves especially simulation of molecules with electron pairs (-NH-, -OH, =O group), triple bonds and other strongly concentrated electrons. Calculation of the overlap repulsive potential is again accelerated by FFT to achieve minimal computational overhead (2-3 minutes) as long as input densities of tip and sample are available.
2. **GPU version:** - Version specially designed for generation of training data for machine learning. Implementation using `pyOpenCL` can parallelize the evaluation of forcefield and relaxation of probe-particle positions over hundreds or thousands of stream-processors of the graphical accelerator. Further speed-up is achieved by using hardware accelerated trilinear interpolation of 3D textures available in most GPUs. This allows simulating 10-100 AFM experiments per second on consumer-grade desktop GPU.
   * GPU version is designed to work in collaboration with machine-learning software for AFM (https://github.com/SINGROUP/ASD-AFM) and use various generators of molecular geometry.    
3. **GUI @ GPU** - The speed of GPU implementation also allows to make interactive GUI where AFM images of molecules can be updated on the fly (<<0.1s) on a common laptop computer while the user is editing molecular geometry or parameters of the tip. This provides an invaluable tool especially to experimentalists trying to identify and interpret the structure and configuration of molecules in experiments on-the-fly while running the experiment. 

#### Other branches

* **master_backup** - Old `master` branch was recently significantly updated and named `main`. For users who miss the old master branch, we provided a backup copy. However, this version is very old and its use is discouraged. If you miss some functionality or are not satisfied with the behavior of current `main` branch please let us know by creating an *issue*.
* **PhotonMap** - implements the latest developments concerning sub-molecular scanning probe combined with Raman spectroscopy (TERS)y and fluorescent spectroscopy (LSTM).
* **complex_tip** - Modification of probe-particle model with 2 particles allows a better fit to experimental results at the cost of additional fitting parameters. 

## Installation & running examples

All development and testing were done on **linux** OS (mostly ubuntu). For the windows version, `docker` version will be prepared soon. In the meantime - several people successfully used probe-particle on windows using [mingw](https://www.mingw-w64.org/).

#### Install & run CPU version

**Requirements:** Python3 (numpy,matplotlib) & C/C++ compiler (g++,make)

##### First run: Graphene with point-charges
 1. clone the repository: `clone https://github.com/Probe-Particle/ProbeParticleModel.git` 
 2. compile the C/C++ modules
    * `cd ProbeParticleModel/Graphene`
    * `make`
 3. Navigate to examples directory `cd ProbeParticleModel/examples/Graphene`this example uses simple (Point-charges + Lennard-Jones)
 4. run the example `./run.sh`
 5. output directory `/examples/Graphene/Q-0.05K0.50/Amp2.0` should contain simulated images with tip charge -0.05e, stiffness 0.5N/m and ossicaltion amplitude 2.0A.

*NOTE:* Python package is designed to automatically recompile the C/C++ automatically, which is convenient for development, so explicit compilation in step #2 maybe not be necessary. see e.g. `cpp_utils.make("PP")` in `ppafm/core.py`

##### Example 2: PTCDA with Hartree potential

1. navigate to `ProbeParticleModel/examples/PTCDA_Hartree`
2. run the example `./run.sh`
   
*NOTE:* Notice that the script `run.sh` downloads and unpack LOCPOT file:
`wget --no-check-certificate "https://www.dropbox.com/s/18eg89l89npll8x/LOCPOT.xsf.zip"`
`unzip LOCPOT.xsf.zip`
this is a large 3D volumetric file which contains Hartree electrostatic potential (in this example computed by VASP) which have to be provided.

##### Example 3: Pyridine with Density-overlap

1. navigate to `ProbeParticleModel/examples/pyridineDensOverlap`
2. run the example `./run.sh`

*NOTE:* Notice that the script `run.sh` downloads and unpacks files `CHGCAR.xsf` & `LOCPOT.xsf` and places them in subdirectories `sample` and `tip`. These are electron density and Hartree potential which need to be provided from DFT calculation (this time from VASP).

#### Install & run GPU GUI

Install prerequisites (Ubuntu):
```sh
sudo apt install git python3-pip python3-pyqt5
pip install matplotlib numpy pyopencl reikna ase
```

Additionally an OpenCL Installable Client Driver (ICD) for your compute device is required:
* Nvidia GPU: comes with the standard Nvidia driver (nvidia-driver-xxx)
* AMD GPU: `sudo apt install mesa-opencl-icd`
* Intel HD Graphics: `sudo apt install intel-opencl-icd`
* CPU: `sudo apt install pocl-opencl-icd`

Run the GUI application:
```sh
./GUI/ppm-gui
```

In order to make the GUI application appear in the system application menu and 'Open with' context menus, link the `ppm-gui` application to a location that is on PATH, e.g. `~/.local/bin`, and install the provided .desktop file. This can be achived by running the following in the repository root:
```bash
ln -s `realpath ./GUI/ppm-gui` $HOME/.local/bin
cp ./GUI/resources/ppm-gui.desktop $HOME/.local/share/applications
```

###### Usage:
* Open a file by clicking `Open File...` at the bottom or provide an input file as a command line argument. The input file can be a .xyz geometry file (possibly with point charges*), a VASP POSCAR or CONTCAR file, an FHI-aims .in file, or a .xsf or .cube Hartree potential file. Loading large files may take some time.
* Changing any number in any input box will automatically update the image. There are also presets for some commonly used tip configurations.
Hover the mouse cursor over any parameter for a tooltip explaining the meaning of the parameter.
* Click anywhere on the image to bring up a plot of the df approach curve for that point in the image.
* Scroll anywhere on the image to zoom the scan window in/out of that spot.
* Click on the `View Geometry` button to show the system geometry in ASE GUI.
* Click on the `Edit Geometry` button to edit the positions, types, and charges of the atoms in the system. Note that for Hartree potential inputs editing charges is disabled and editing the geometry only affects the Lennard-Jones force field.
* Click on the `View Forcefield` button to view different components of the force field. Note that the forcefield box size is inferred automatically from the scan size and is bigger than the scan size. Take into account the probe particle equilibrium distance when comparing the reported z-coordinates between the forcefield and the df image.
* Click on the `Edit Forcefield` button to edit the per-species parameters of the Lennard-Jones forcefield.
* Save the current image or df data by clicking the `Save Image...` or `Save df...` buttons at the bottom.
* In case there are multiple OpenCL devices installed on the system, use the `-l` or `--list-devices` option to list available devices and choose the device using the `-d` or `--device` option with the device platform number as the argument.

*Note that while input files without charges work, depending on the system, the resulting image may be significantly different from an image with electrostatics, and therefore may not be representative of reality. If no electrostatics are included, this is indicated in the title of the image.

#### Run GPU generator for machine learning

* `examples/CorrectionLoopGraphene` use GPU accelerated PPM to iteratively improve the estimate of molecular geometry by comparing simulated AFM images with reference. This is work-in-progress. Currently, modification of estimate geometry is random (Monte-Carlo), while later we plan to develop a more clever (e.g. Machine-Learned) heuristic for more efficient improvment.
* `examples/Generator` quickly generates a batch of simulated AFM images (resp. 3D data stacks) which can be further used for machine learning. Especially in connection with (https://github.com/SINGROUP/ASD-AFM).

### Notable publications using Probe Particle Model

* [Prokop Hapala, Georgy Kichin, Christian Wagner, F. Stefan Tautz, Ruslan Temirov, and Pavel Jelínek, Mechanism of high-resolution STM/AFM imaging with functionalized tips, Phys. Rev. B 90, 085421 – Published 19 August 2014](http://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.085421)
* [Prokop Hapala, Ruslan Temirov, F. Stefan Tautz, and Pavel Jelínek, Origin of High-Resolution IETS-STM Images of Organic Molecules with Functionalized Tips, Phys. Rev. Lett. 113, 226101 – Published 25 November 2014,](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.226101) 

### License
MIT

