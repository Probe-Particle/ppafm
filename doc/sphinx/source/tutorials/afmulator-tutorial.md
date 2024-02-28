# GPU-accelerated AFM simulations

The command-line interface to ppafm can only utilize the CPU.
The simulations can be sped-up by at least a couple of orders of magnitude by utilizing a graphics processing unit (GPU).
In order to make use of the GPU acceleration in ppafm, install it with `pip` using the `[opencl]` option, as detailed on the  [installation page on the wiki](https://github.com/Probe-Particle/ppafm/wiki/Install-ppafm).

The tutorial here concerns how to run GPU-accelerated AFM simulations programmatically via the Python API, which is most useful when running a large number of simulations in a batch.
If you are looking to study an individual system and experiment with the simulation parameters, the [graphical user interface](https://github.com/Probe-Particle/ppafm/wiki/PPAFM-GUI)
may be better suited for that purpose.
This tutorial also assumes that you are already familiar with the basic of the structure of the ppafm simulation. If not, take a look at the wiki pages explaining the [basics of the simulation](https://github.com/Probe-Particle/ppafm/wiki#probe-particle-model) and the [different force field models](https://github.com/Probe-Particle/ppafm/wiki/Forces).

## The AFMulator API

The main interface for accessing the GPU acceleration in `ppafm` is the {class}`.AFMulator` class. The way to run a simulation is to first create an instance of the class, with arguments that specify the simulation parameters, and then call the created object with an input molecular geometry. For example:
```python
from ppafm.io import loadXYZ
from ppafm.ocl.AFMulator import AFMulator

# Load sample coordinates (xyzs), atomic numbers (Zs), and charges (qs)
xyzs, Zs, qs, _ = loadXYZ("./Gr6x6N3hole.xyz")

# Create an instance of the simulator
afmulator = AFMulator(
    scan_dim=(201, 201, 40),
    scan_window=((0.0, -10.0, 5.0), (20.0, -10.0, 9.0)),
    iZPP=8,
    df_steps=10,
    tipR0=[0, 0, 4]
)

# Run the simulation and plot the resulting images
afm_images = afmulator(xyzs, Zs, qs, plot_to_dir="./afm_ocl")
```

At the first step we simply load a sample molecule to the memory (using here the [graphene example](https://github.com/Probe-Particle/ppafm/tree/main/examples/Graphene)):
```python
xyzs, Zs, qs, _ = loadXYZ("./Gr6x6N3hole.xyz")
print(xyzs.shape, Zs.shape, qs.shape) # (71, 3) (71,) (71,)
```
Here we get the three main inputs that we need for the simulation as `numpy` arrays: the atomic coordinates (`xyzs`), the atomic numbers (`Zs`), and partial charges for each atom (`qs`).Instead of partial charges, the Hartree potential for the sample can be used as well (see [](#hartree-electrostatics)).

The second step creates the simulator object with some of the basic parameters:
```python
afmulator = AFMulator(
    pixPerAngstrome=10,
    scan_dim=(201, 201, 40),
    scan_window=((0.0, -10.0, 5.0), (20.0, 10.0, 9.0)),
    iZPP=8,
    df_steps=10,
    tipR0=[0, 0, 4]
)
```
We set here the following parameters:
- `pixPerAngstrome` sets the density of points in the force-field grid. The default value of 10 is usually adequate, but is some cases a higher value can lead to slighly more accurate results, but at the expense of higher video memory usage. Try setting this lower if you get memory errors on devices with a low amount of video memory.
- `scan_dim` sets the number points in the x, y, and z directions in the scan region, and the `scan_window` sets the physical size of the scan region as a tuple with the start and end points (opposing corners) of the region in units of Ångströms.
- `iZPP` sets the atomic number of the probe particle, which affects the used Lennard-Jones parameters.
- The amplitude of the oscillation is set by the `df_steps` parameter, which specifies how many steps in the z-direction are used in the $F_\mathrm{z} \rightarrow \Delta f$ conversion. The step size in z-direction is determined as `(scan_window[1][2] - scan_window[0][2]) / scan_dim`, so in this case the amplitude is `10 * (9.0Å - 5.0Å) / 40 = 1.0Å`, and the final number of constant-height images is `scan_dim[2] - df_steps + 1 = 40 - 30 + 1 = 31`.
- The equilibrium position of the probe particle is set by the `tipR0` parameter. Note that the `tipR0` is added to the `scan_window` coordinates, and the z-direction in this parameter is reversed, so that at closest approach the equilibrium position of the probe particle is at `5.0Å - 4.0Å = 1.0Å`.

There are many other arguments that can be specified, which can be found in the documentation page for the {class}`.AFMulator` class.

```{note}
Note that there is a slight inconsistency here between the xy- and the z-directions in how the scan region is determined. In the xy-direction, the scan region includes both the start and the end points, but in the z-direction the start point is not included, so the step size in the above example is 0.1Å in all directions, despite the `scan_dim` having an odd number of points in the xy-directions.
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
The three arguments `xyzs`, `Zs`, and `qs` are mandatory for every simulation, and additional inputs may be required depending on the type of force field model used (see the following sections). The optional argument `plot_to_dir` will plot the resulting images into the specified directory. We can also get the image array as the return value in order to plot the images manually or to do some other analysis:
```python
import matplotlib.pyplot as plt
afm_images = afmulator(xyzs, Zs, qs)
print(afm_images.shape) # (201, 201, 31)
for i in range(afm_images.shape[2]):
    plt.imshow(afm_images[:, :, i].T, origin='lower', cmap='afmhot')
    plt.savefig(f'afm_{i}.png')
    plt.close()
```
The returned images are saved in a 3D-array corresponding to all the points in the scan. The array axes correspond to the x-, y-, and z-axes. The z-axis is ordered so that the element at index 0 is the highest z-coordinate.

```{note}
Note that the xyz-ordering of the axes here is different from the matrix ij-ordering more commonly used with image arrays. This can be seen in the plotting step above where we transpose the array with `.T` and specify `origin='lower'`.
```

### Lennard-Jones

(hartree-electrostatics)=
### Hartree electrostatics

### Full-density based model
