# pyLightSTM: Simulation of High-Resolution Molecular Imaging with Tip-Enhanced Electro-Luminescence

## Introduction

LightSTM is an advanced scanning probe-microscopy technique that combines Scanning Tunneling Microscopy (STM) with optical spectroscopy to achieve high-resolution imaging of molecules. The method leverages the enhancement of optical signals through the cavity formed by the metallic STM tip, providing both spatial and spectroscopic information about molecular systems. Simulation of the signal therefore comprise simpulation of both these processes (1) tunneling rate and (2) coupling of molecular fluorescence to optical cavity.

## Theoretical Background

### 1. STM Tunneling Signal

The tunneling current modulates the luminescence by controlling the population of excited states. The tunneling $t_{STM}$ rate is given by:

$$ t_{STM}(R_{tip}) = \int |\psi_{tip}(r - R_{tip})| \cdot |\psi_{mol}(r)| dr $$

where:
- $\psi_{tip}$ is the tip wavefunction (approximated by exponential decay)
- $\psi_{mol}$ is the molecular orbital wavefunction
- The tunneling decay is modeled as: $\psi_{tip}(r) \propto e^{-\beta |r|}$

### 2. Optical Cavity Coupling

The optical coupling between molecular transitions and the tip cavity is described by the interaction between the transition density of the molecule and the electromagnetic field of the tip. The optical transition rate is given by:

$$ O_{opt}(R_{tip}) = \int \rho_{trans}(r) \cdot \phi_{tip}(r - R_{tip}) dr $$

where:
- $\rho_{trans}$ is the transition density between initial and final molecular states
- $\phi_{tip}$ is the tip's cavity field
- $R_{tip}$ is the tip position
- $r$ is the integration variable

The tip field is modeled using multipole expansion:

$$ \phi_{tip}(r) = \sum_{\ell m} c_{\ell m} Y_{\ell m}(\theta, \phi) \frac{1}{\sqrt{r^2 + \sigma^2}} $$

where $Y_{\ell m}$ are spherical harmonics and $c_{\ell m}$ are multipole coefficients.

### 3. FFT-Based Acceleration

Both the optical coupling and tunneling integrals are efficiently computed using Fast Fourier Transform (FFT), as they can be expressed as convolutions:

$$ M(R_{tip}) = \int f_{sample}(r) \cdot g_{tip}(r + R_{tip}) dr = \mathcal{F}^{-1}[\mathcal{F}(f_{sample}) \cdot \mathcal{F}(g_{tip})] $$

### 4. Signal Combination and Image Formation

The final LightSTM signal is formed by combining the optical coupling and tunneling contributions. The process works as follows:

1. **STM Signal Generation** (`makeSTMmap`):
   - Calculates tunneling probability using molecular wavefunctions
   - Produces current map proportional to square of tunelling rate: $I_{STM} = |t_{STM}|^2$
   - This modulates the excitation probability of molecules

2. **Optical Signal Generation** (`makePhotonMap`):
   - Computes optical coupling using transition densities
   - Produces photon intensity map: $I_{opt} = |O_{opt}|^2$
   - Represents the radiative decay probability

3. **Combined Signal**:
   - The total signal is proportional to both the excitation (STM) and decay (optical) probabilities
   - Final intensity at each point: $I_{total}(R_{tip}) \propto I_{STM}(R_{tip}) \cdot I_{opt}(R_{tip})$
   - This reflects that luminescence requires both efficient excitation and radiative decay

## Implementation Details

### Core Components

1. **Tip Field Generation** (`makeTipField` in `photo.py`)
   - Constructs the electromagnetic field of the tip using multipole expansion
   - Supports both 2D and 3D field calculations
   - Parameters include tip height, decay factor, and multipole coefficients

2. **Exciton System Solver** (`solveExcitonSystem` in `photo.py`)
   - Implements coupled exciton calculations
   - Assembles and diagonalizes the Hamiltonian matrix
   - Computes eigenvalues and eigenvectors for the coupled system

3. **Photon Map Generation** (`makePhotonMap` in `photonMap.py`)
   - Combines optical coupling and tunneling calculations
   - Projects transition densities onto a common grid
   - Applies FFT-based convolution for efficient computation

### Key Functions and Dependencies

1. **Grid Operations** (`GridUtils.py`)
   - `makeTransformMat`: Creates transformation matrices for molecular orientations
   - `evalMultipole`: Computes multipole coefficients for molecular transitions
   - Handles grid manipulations and coordinate transformations

2. **FFT Operations** (`fieldFFT.py`)
   - Implements efficient Fourier transforms for convolution calculations
   - Manages periodic boundary conditions and grid alignments

3. **Photon Mapping** (`photo.py`)
   - `photonMap2D_stamp`, `photonMap3D_stamp`: Generate photon maps in 2D and 3D
   - `convFFT`: Performs convolution using FFT
   - `prepareRhoTransForCoumpling`: Prepares transition densities for coupling calculations

### Workflow

1. **Initialization**
   - Load molecular geometries and wavefunctions
   - Set up computational grid and tip parameters
   - Initialize transition densities

2. **Exciton System Solution**
   - Assemble Hamiltonian matrix
   - Solve for eigenstates and energies
   - Compute coupling coefficients

3. **Image Generation**
   - Calculate tip fields and tunneling probabilities
   - Perform FFT-based convolutions
   - Generate final photon maps

## Function Reference

### Grid and Coordinate Operations (`photo.py`)

1. `makeTransformMat(ns, lvec, angle=0.0, rot=None)`
   - Creates transformation matrices for molecular orientations
   - Parameters: grid dimensions (`ns`), lattice vectors (`lvec`), rotation angle
   - Returns: transformation matrix for coordinate mapping

2. `getMGrid2D(nDim, dd)`, `getMGrid3D(nDim, dd)`
   - Generate coordinate grids for calculations
   - Parameters: dimensions (`nDim`), grid spacing (`dd`)
   - Returns: coordinate arrays (X, Y, Z) and grid shifts

3. `makeTipField(sh, dd, z0=10.0, sigma=1.0, multipole_dict={'s':1.0}, b3D=False, bSTM=False, beta=1.0)`
   - Constructs tip field using multipole expansion
   - Supports both STM and optical field calculations
   - Parameters: shape (`sh`), spacing (`dd`), tip height (`z0`), decay parameters

### Exciton System Solver (`photo.py`)

1. `solveExcitonSystem(rhoTranss, lvecs, poss, rots, nSub=None, byCenter=False, Ediags=1.0)`
   - Main solver for coupled exciton systems
   - Implements quantum mechanical coupling between molecules
   - Returns eigenvalues and eigenvectors of the system

2. `assembleExcitonHamiltonian(rhos, poss, latMats, Ediags, byCenter=False)`
   - Constructs Hamiltonian matrix for exciton system
   - Includes position-dependent coupling terms
   - Used internally by `solveExcitonSystem`

### Photon Mapping (`photonMap.py`)

1. `makePhotonMap(S, ipl, coefs, Vtip, dd_canv, byCenter=False, bDebugXsf=False)`
   - Generates optical coupling maps
   - Supports both 2D and 3D calculations
   - Parameters: system (`S`), eigenvector index (`ipl`), coefficients, tip potential

2. `makeSTMmap(S, coefs, wfTip, dd_canv, byCenter=False)`
   - Calculates tunneling probability maps
   - Combines molecular and tip wavefunctions
   - Returns intensity map of tunneling current

### Utility Functions

1. `loadCubeFiles(S0)`
   - Loads molecular orbital data from Gaussian cube files
   - Handles both wavefunctions and transition densities

2. `loadDicts(fname, convertor=float)`
   - Loads parameter dictionaries from configuration files
   - Used for tip and molecular specifications

3. `plotPhotonMap(system, ipl, ncomb, nvs, byCenter=False, fname=None, dd=None)`
   - Visualizes calculated photon maps
   - Supports various output formats and plotting options

## References

1. Time Dependent Quantum Mechanics and Spectroscopy (Tokmakoff) - Chapter 15: Energy and Charge Transfer
2. Excitons in Molecular Aggregates ([LibreTexts](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Book%3A_Time_Dependent_Quantum_Mechanics_and_Spectroscopy_(Tokmakoff)/15%3A_Energy_and_Charge_Transfer/15.03%3A_Excitons_in_Molecular_Aggregates))
