# GPU-accelerated AFM simulations

The command-line interface to ppafm can only utilize the CPU.
The simulations can be sped-up by at least a couple of orders of magnitude by utilizing a graphics processing unit (GPU).
In order to make use of the GPU acceleration in ppafm, install it with `pip` using the `[opencl]` option, as detailed on the  [installation page on the wiki](https://github.com/Probe-Particle/ppafm/wiki/Install-ppafm).

The tutorial here concerns how to run GPU-accelerated AFM simulations programmatically via the Python API, which is most useful when running a large number of simulations in a batch.
If you are looking to study an individual system and experiment with the simulation parameters, the [graphical user interface](https://github.com/Probe-Particle/ppafm/wiki/PPAFM-GUI)
may be better suited for that purpose.
This tutorial also assumes that you are already familiar with the basic of the structure of the ppafm simulation. If not, take a look at the wiki pages explaining the [basics of the simulation](https://github.com/Probe-Particle/ppafm/wiki#probe-particle-model) and the [different force field models](https://github.com/Probe-Particle/ppafm/wiki/Forces).

## The AFMulator

The main interface for accessing the GPU acceleration in `ppafm` is the {class}`.AFMulator` class.
The way to run a simulation is to first create an instance of the class, with arguments that specify the simulation parameters, and then call the created object with an input molecular geometry.
For example:
```python
from ppafm.io import loadXYZ
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env

# Initialize an OpenCL environment. You can change i_platform to select the device to use.
init_env(i_platform=0)

# Load sample coordinates (xyzs), atomic numbers (Zs), and charges (qs)
xyzs, Zs, qs, _ = loadXYZ("./Gr6x6N3hole.xyz")

# Create an instance of the simulator
afmulator = AFMulator(
    scan_dim=(201, 201, 40),
    scan_window=((0.0, -10.0, 5.0), (20.0, 10.0, 9.0)),
    iZPP=8,
    df_steps=10,
    tipR0=[0.0, 0.0, 3.0]
)

# Run the simulation and plot the resulting images
afm_images = afmulator(xyzs, Zs, qs, plot_to_dir="./afm_ocl")
```

```{note}
The underlying implementation of `AFMulator` is based on OpenCL and requires an OpenCL-enabled compute device to be installed on the system with supported drivers.
By default when creating an `AFMulator` instance, the first available device is used.
On systems with multiple compatible devices, the {func}`.init_env` function can be used to select a specific device to use.
An easy way to find the number of the correct device is to list all of the devices usable by `ppafm` by running `ppafm-gui --list-devices` on the command line.
```

At the first step we simply load a sample molecule to the memory (using here the [graphene example](https://github.com/Probe-Particle/ppafm/tree/main/examples/Graphene)):
```python
xyzs, Zs, qs, _ = loadXYZ("./Gr6x6N3hole.xyz")
print(xyzs.shape, Zs.shape, qs.shape) # (71, 3) (71,) (71,)
```
Here we get the three main inputs that we need for the simulation as `numpy` arrays: the atomic coordinates (`xyzs`), the atomic numbers (`Zs`), and partial charges for each atom (`qs`).
Instead of partial charges, the Hartree potential for the sample can be used as well (see [](#hartree-electrostatics)).

The second step creates the simulator object with some of the basic parameters:
```python
afmulator = AFMulator(
    pixPerAngstrome=10,
    scan_dim=(201, 201, 40),
    scan_window=((0.0, -10.0, 5.0), (20.0, 10.0, 9.0)),
    iZPP=8,
    df_steps=10,
    tipR0=[0.0, 0.0, 3.0]
)
```
We set here the following parameters:
- `pixPerAngstrome` sets the density of points in the force-field grid.
The default value of 10 is usually adequate, but in some cases a higher value can lead to slighly more accurate results, but at the expense of higher video memory usage.
Try setting this lower if you get memory errors on devices with a low amount of video memory.
- `scan_dim` sets the number points in the x, y, and z directions in the scan region, and the `scan_window` sets the physical size of the scan region as a tuple with the start and end points (opposing corners) of the region in units of Ångströms.
- `iZPP` sets the atomic number of the probe particle, which affects the used Lennard-Jones parameters.
- The amplitude of the oscillation is set by the `df_steps` parameter, which specifies how many steps in the z-direction are used in the $F_\mathrm{z} \rightarrow \Delta f$ conversion.
The step size in z-direction is determined as `(scan_window[1][2] - scan_window[0][2]) / scan_dim`, so in this case the amplitude is `10 * (9.0Å - 5.0Å) / 40 = 1.0Å`, and the final number of constant-height images is `scan_dim[2] - df_steps + 1 = 40 - 30 + 1 = 31`.
- The equilibrium position of the probe particle is set by the `tipR0` parameter.
Note that the `tipR0` is added to the `scan_window` coordinates, and the z-direction in this parameter is reversed, so that at closest approach the equilibrium position of the probe particle is at `5.0Å - 3.0Å = 2.0Å`.

There are many other arguments that can be specified, which can be found in the documentation page for the {class}`.AFMulator` class.

```{note}
Note that there is a slight inconsistency here between the xy- and the z-directions in how the scan region is determined.
In the xy-direction, the scan region includes both the start and the end points, but in the z-direction the start point is not included, so the step size in the above example is 0.1Å in all directions, despite the `scan_dim` having an odd number of points in the xy-directions.
```

````{tip}
If you have a `params.ini` file from the `ppafm` CLI, you may also construct an `AFMulator` instance from such a file:
```python
afmulator = AFMulator.from_params("./params.ini")
```
````

The last step runs the simulation:
```python
afmulator(xyzs, Zs, qs, plot_to_dir="./afm_ocl")
```
The three arguments `xyzs`, `Zs`, and `qs` are mandatory for every simulation, and additional inputs may be required depending on the type of force field model used (see the following sections).
The optional argument `plot_to_dir` will plot the resulting images into the specified directory.
We can also get the image array as the return value in order to plot the images manually or to do some other analysis:
```python
import matplotlib.pyplot as plt
afm_images = afmulator(xyzs, Zs, qs)
print(afm_images.shape) # (201, 201, 31)
for i in range(afm_images.shape[2]):
    plt.imshow(afm_images[:, :, i].T, origin='lower', cmap='afmhot')
    plt.savefig(f'afm_{i}.png')
    plt.close()
```
The returned images are saved in a 3D-array corresponding to all the points in the scan.
The array axes correspond to the x-, y-, and z-axes.
The z-axis is ordered so that the element at index 0 is the highest z-coordinate.

```{note}
Note that the xyz-ordering of the axes here is different from the matrix ij-ordering more commonly used with image arrays.
This can be seen in the plotting step above where we transpose the array with `.T` and specify `origin='lower'`.
```

The type of force field used in the simulation is decided by the input arguments when calling the `AFMulator` object.
The following sections discuss how to use the different force fields.

### Lennard-Jones with point-charge electrostatics

The Lennard-Jones force field is the default force field, which is used when only the `xyzs`, `Zs`, and `qs` arguments are given without the sample electron density `rho_sample`:
```python
afm_images = afmulator(xyzs, Zs, qs, rho_sample=None) # rho_sample=None is the default, but is shown explicitly here for clarity
```
The type of the `qs` argument decides what kind of electrostatics is used.
In order to use point-charge electrostatics, `qs` needs to be a `numpy` array of charges for each atom in the system.
Typically the charges would be stored in a .xyz file and can be loaded using the {func}`.loadXYZ` function as shown in the examples above.
The charges can come from, for example, Hirshfeld or Mulliken charge-partition analysis.

The simulation can also be run without the electrostatic contribution.
If the input .xyz file does not contain point charges for the atoms, then `loadXYZ` will simply return an array of zeros for `qs`:
```python
import numpy as np
xyzs, Zs, qs, _ = loadXYZ("./molecule_without_charges.xyz")
print(np.allclose(qs), 0) # True
afm_images = afmulator(xyzs, Zs, qs) # No electrostatics
```
The `qs` argument can also be explicitly set to `None` to skip the electrostatics calculation:
```python
afm_images = afmulator(xyzs, Zs, qs=None) # No electrostatics
```

We also need to set the charge of the probe particle.
This is done during the construction of the AFMulator instance:
```python
afmulator = AFMulator(
    ...
    iZPP=8, # O
    Qs=[-10, 20, -10, 0],
    QZs=[0.07, 0.0, -0.07, 0]
)
```
The charge is set by specifying the magnitude (`Qs`) and the positions (`QZs`) of up to four point charges on the z-axis, with respect to the position of the probe particle.
By using multiple charges it is possible to create multipole moments.
In the above example we create a quadrupole charge distribution typical for a CO tip by placing three charges in a `[q, -2q, q]` pattern equidistantly.
The quadrupole moment here comes to $-10 e \times (0.07Å)^2 \approx -0.05 e \times Å^2$.
For other tips, we may want to use a different charge distribution.
For example, for a Xe tip a monopole charge is a better model:
```python
afmulator = AFMulator(
    ...
    iZPP=54, # Xe
    Qs=[0.3, 0.0, 0.0, 0.0],
    QZs=[0.0, 0.0, 0.0, 0.0]
)
```

(hartree-electrostatics)=
### Lennard-Jones with Hartree electrostatics

A more accurate calculation of the electrostatic force can be done using the Hartree potential of the sample, typically calculated by density-functional theory (DFT) codes such as VASP or FHI-aims.
We can use a Hartree potential saved in a .xsf or .cube file by loading it into an instance of the {class}`.HartreePotential` class:
```python
from ppafm.ocl.AFMulator import HartreePotential
potential, xyzs, Zs = HartreePotential.from_file("./LOCPOT.xsf", scale=-1.0)  # scale=-1.0 for correct units of potential (V) instead of energy (eV)
```
Since the file also contains the geometry of the system, in addition to the `potential`, we also get the `xyzs` and `Zs` when loading the file.
In order to use the loaded potential, we simply substitute it for the `qs` argument when running the simulation:
```python
afm_images = afmulator(xyzs, Zs, qs=potential)
```

```{note}
The `AFMulator` expects the potential to be in units of volts, but most DFT codes output the potential in units of energy, typically in eV or in Hartree.
When loading files, `ppafm` currently always assumes that .xsf files are in units of eV, and .cube files are in units of Hartree which immediately get converted to eV.
In order to convert the eV to V, we simply multiply by -1 (corresponding to division by the electron charge, -e), hence the `scale=-1.0` when loading the Hartree potential above.
```

Again, we need to set the charge distribution of the probe particle as well.
When using the Hartree electrostatics, this is done by setting the `rho` and `sigma` parameters:
```python
afmulator = AFMulator(
    ...
    rho={"dz2": -0.1},
    sigma=0.71
)
```
Here, `rho` specifies the charge distribution as a dictionary of orbitals with magnitudes, and `sigma` sets the width of the distribution in Ångströms.
Above, the quadrupole `dz2` orbital is used, but we can also use for example the monopole `s` orbital, the dipole `pz` orbital, or a linear combination of multiple of these, for example `rho={"dz2": -0.05, "s": -0.1}`.

The analytical multipole charge distribution works well in most cases, but one can also load a DFT-calculated charge distribution from a file:
```python
from ppafm.ocl.AFMulator import TipDensity
rho_tip, _, _ = TipDensity.from_file("CO_delta_density.xsf")
afmulator = AFMulator(
    ...
    rho=rho_tip,
)
```
The value for `sigma` is in this case ignored.

```{warning}
Loading tip densities works correctly only for .xsf file for the moment, because the values in .cube files are wrongly automatically scaled ([Github issue](https://github.com/Probe-Particle/ppafm/issues/190)).
```

For an example of using Hartree potential in a simulation, see the [PTCDA example](https://github.com/Probe-Particle/ppafm/blob/main/examples/PTCDA_Hartree/run_gpu.py).

### Full-density-based model

The full-density-based model (FDBM) can provide a better approximation of the Pauli repulsion than Lennard-Jones.
In order to use the FDBM, more input files and parameters are required:
```python
from ppafm.ocl.AFMulator import AFMulator, HartreePotential, ElectronDensity, TipDensity
pot, xyzs, Zs = HartreePotential.from_file("LOCPOT.xsf", scale=-1) # Sample Hartree potential
rho_sample, _, _ = ElectronDensity.from_file("CHGCAR.xsf") # Sample electron density
rho_tip, xyzs_tip, Zs_tip = TipDensity.from_file("density_CO.xsf") # Tip electron density
rho_tip_delta, _, _ = TipDensity.from_file("CO_delta_density.xsf") # Tip electron delta-density

afmulator = AFMulator(
    ...
    rho=rho_tip,
    rho_delta=rho_tip_delta
    A_pauli=12.0,
    B_pauli=1.2,
    d3_params="PBE",
)

afm_images = afmulator(xyzs, Zs, qs=pot, rho_sample=rho_sample)
```
Above, we first load the input files.
In addition to the Hartree potential we also load the sample electron density `rho_sample`, and two densities for the tip, the full electron density `rho_tip`, and a delta density `rho_tip_delta`.
When constructing the `AFMulator`, we use both of the densities.
In this case the `rho=rho_tip` is used for the Pauli density overlap intergral, and the `rho_delta=rho_tip_delta` is used in the electrostatics calculation as in the previous section.
The delta density can be separately calculated and loaded from a file as above, or it can be estimated by subtracting the valence electrons from the full electron density:
```python
rho_tip_delta = rho_tip.subCores(xyzs_tip, Zs_tip, Rcore=0.7)
```

We also specify the two parameters of the Pauli overlap integral, the prefactor `A_pauli` and the exponent `B_pauli`, as well as the DFT-D3 parameters based on the functional name via `d3_params`.
In this case we suppose that the electron densities were calculated using the `PBE` functional, so we use the corresponding parameters.
Other predefined parameter sets can be found in the description of the method {meth}`.add_dftd3`.

Finally, when running the calculation we use the loaded `rho_sample`, which causes the FDBM to be used instead of Lennard-Jones:
```python
afm_images = afmulator(xyzs, Zs, qs=pot, rho_sample=rho_sample)
```

For examples of using the FDBM, see the [pyridine example](https://github.com/Probe-Particle/ppafm/blob/main/examples/pyridineDensOverlap/run_gpu.py), as well as another [more advanced example](https://github.com/Probe-Particle/ppafm/blob/main/examples/paper_figure/run_simulation.py) where multiple simulations are performed using all of the different force field models described here.

```{warning}
Electron densities from FHI-aims are known to not work well with the FDBM. This simulation model is generally less explored compared to the other ones, so expect more problems. Known working configurations have used densities calculated with VASP.
```
