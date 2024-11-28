# ChargeRings: Simulation of Molecular Charging and STM Imaging with Electrostatic Interactions

## Introduction

ChargeRings is a method for simulating scanning tunneling microscopy (STM) measurements of molecular systems where charging effects and electrostatic interactions play a significant role. The method considers a system of molecular sites that can be charged/discharged based on their energy levels, mutual Coulomb interactions, and interaction with an STM tip.

## Theoretical Background

### 1. Total Energy of the System

The total energy of the system is given by:

\[
U_{\text{total}} = \sum_{i=1}^N \left( \frac{E_i Q_i}{2} + \frac{Q_i Q_{\text{tip}}}{4 \pi \epsilon_0 r_{i\text{tip}}} - \mu Q_i \right) + \sum_{i=1}^N \sum_{j=i+1}^N \frac{Q_i Q_j}{4 \pi \epsilon_0 r_{ij}}
\]

where:
- $E_i$ is the on-site energy of site i
- $Q_i$ is the charge at site i
- $Q_{\text{tip}}$ is the tip charge
- $r_{i\text{tip}}$ is the distance between site i and the tip
- $r_{ij}$ is the distance between sites i and j
- $\mu$ is the chemical potential (Fermi level)
- $\epsilon_0$ is the vacuum permittivity

### 2. Charging Force

The variational derivative of the total energy with respect to site charges gives the charging force:

\[
\frac{\delta U_{\text{total}}}{\delta Q_i} = \frac{E_i}{2} + \frac{Q_{\text{tip}}}{4 \pi \epsilon_0 r_{i\text{tip}}} + \sum_{\substack{j=1 \\ j \neq i}}^N \frac{Q_j}{4 \pi \epsilon_0 r_{ij}} - \mu
\]

This force determines whether a site tends to gain or lose charge.

### 3. Multipole Interactions

The electrostatic interactions can include higher-order multipole moments (monopole, dipole, quadrupole) for more accurate representation of molecular charge distributions. The multipole energy is computed as:

\[
E_{\text{multipole}} = \frac{Q_0}{r} + \frac{\mathbf{p} \cdot \mathbf{r}}{r^3} + \frac{1}{2} \sum_{ij} Q_{ij} \frac{3x_i x_j - r^2\delta_{ij}}{r^5}
\]

where $Q_0$ is the monopole, $\mathbf{p}$ is the dipole moment, and $Q_{ij}$ is the quadrupole tensor.

## Implementation Details

### Core Components

1. **RingParams Structure**
   - Holds system parameters including:
     - Number of molecular sites
     - Site positions and orientations
     - Multipole moments
     - Energy levels
     - Fermi level
     - Coulomb coupling strength
     - Temperature

2. **Coupling Matrix Generation** (`makeCouplingMatrix`)
   - Constructs the Hamiltonian matrix including:
     - On-site energies
     - Tip-site interactions
     - Site-site Coulomb interactions
   - Handles multipole interactions through rotation matrices

3. **Site Occupation Solvers**
   - `optimizeSiteOccupancy`: Gradient descent optimization of site charges
   - `boltzmanSiteOccupancy`: Statistical mechanics approach using Boltzmann distribution
   - Both methods minimize the total energy while respecting physical constraints (0 ≤ Q ≤ 1)

4. **STM Signal Generation**
   - `getSTM`: Computes tunneling current based on:
     - Site occupations
     - Tip-site distances
     - Decay constants
     - Temperature effects

### Key Functions

1. **Initialization** (`initRingParams`)
   - Sets up system parameters
   - Allocates memory for position arrays and multipole moments
   - Initializes energy levels and coupling constants

2. **Energy Calculations**
   - `getChargingForce`: Computes forces on charges
   - `Emultipole`: Evaluates multipole interaction energies
   - Includes on-site Coulomb repulsion for multiple charges

3. **Dynamics** (`moveGD`, `moveMD`)
   - Gradient descent optimization
   - Molecular dynamics with damping
   - Enforces charge constraints

4. **Green's Function Methods**
   - `solveHamiltonian`: Eigenvalue problem solver
   - `computeGreensFunction`: Non-equilibrium Green's function calculations
   - Handles many-body effects in transport

### Workflow

1. **System Setup**
   - Define molecular geometry
   - Set energy levels and coupling parameters
   - Initialize tip parameters

2. **Charge Optimization**
   - Compute Coulomb interactions
   - Optimize site charges
   - Include temperature effects if needed

3. **STM Signal**
   - Calculate tunneling currents
   - Generate STM maps
   - Include multipole effects in imaging

## Usage Notes

1. **Temperature Effects**
   - Use `boltzmanSiteOccupancy` for finite temperature
   - Set appropriate energy scales relative to kT

2. **Convergence**
   - Adjust optimization parameters (dt, damping) for stability
   - Monitor energy convergence
   - Check charge conservation

3. **Multipole Implementation**
   - Provide accurate multipole moments
   - Ensure proper orientation matrices
   - Consider computational cost vs accuracy
