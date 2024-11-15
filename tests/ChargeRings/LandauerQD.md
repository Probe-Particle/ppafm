# Modeling Electron Transport in a Three-Quantum-Dot System using the Landauer-Büttiker Formalism

This work presents a computational model for simulating electron transport through a triangular arrangement of three quantum dots (QDs) coupled to a metallic substrate and probed by a scanning tunneling microscope (STM) tip. The model employs the Landauer-Büttiker formalism, a well-established framework for describing coherent quantum transport.

The system is represented by a 5x5 Hamiltonian matrix, encompassing the STM tip, the substrate, and the three QDs. The tip and substrate are treated as electron reservoirs, while the QDs are modeled as discrete energy levels. The Hamiltonian incorporates the following key interactions:

1. **Tip-QD Coupling:**  Tunneling between the STM tip and each QD is modeled using an exponential decay dependence on the tip-QD distance ($t_{ti} = t_0 e^{-\beta |r_{tip} - r_i|}$), reflecting the localized nature of the tunneling process. Direct tip-substrate tunneling is neglected in this model.

2. **QD-Substrate Coupling:**  A constant coupling strength ($t_s$) is assumed between each QD and the substrate, representing the strong hybridization between the QDs and the underlying metallic surface.

3. **Inter-QD Coulomb Interaction:**  A constant Coulomb interaction strength ($K$) is considered between each pair of QDs, accounting for electrostatic interactions within the triangular arrangement.

The central quantity of interest is the transmission probability $T(E)$, which describes the probability of an electron tunneling from the tip to the substrate at a given energy $E$. $T(E)$ is calculated using the Green's function formalism:

$T(E) = \text{Tr}[\Gamma_t G^r(E) \Gamma_s G^a(E)]$

where $G^r(E)$ and $G^a(E)$ are the retarded and advanced Green's functions of the substrate+QD system, and $\Gamma_t$ and $\Gamma_s$ are the broadening matrices describing the coupling to the tip and substrate, respectively. These broadening matrices are calculated within the wide-band limit approximation.

The model allows for the investigation of interference effects in the electron transport, including Fano resonances, which arise from the interplay between different tunneling pathways. By varying the tip position and the energy, the model can simulate STM dI/dV maps and spectra, providing insights into the electronic structure and charge distribution within the 3QD system. Future work could explore the inclusion of tip-induced level shifting and more sophisticated models for the QD-substrate interaction. This model provides a valuable tool for understanding and interpreting experimental STM data on coupled quantum dot systems.


# LandauerQDs Class Documentation

This class implements the Landauer-Büttiker formalism to calculate the transmission probability through a system of three quantum dots (QDs) coupled to a substrate and probed by an STM tip.

## Class Definition

```python
class LandauerQDs:
    def __init__(self, qd_positions, K, decay, on_site_energies, t_substrate, E_sub=0.0, t0=0.1, eta=0.01):
       # ... (Initialization code as in the previous response)
```

## Attributes

* `qd_positions` (np.ndarray): A 3x3 array containing the positions of the three QDs: $[[x_1, y_1, z_1], [x_2, y_2, z_2], [x_3, y_3, z_3]]$.
* `K` (float): Coulomb interaction strength between the QDs (assumed to be the same for all pairs): $K_{ij} = K$ for $i \neq j$.
* `decay` (float): Decay constant $\beta$ for the tip-QD coupling: $t_{ti} = t_0 e^{-\beta |r_{tip} - r_i|}$.
* `on_site_energies` (np.ndarray): A 1D array of on-site energies $\epsilon_i$ for the QDs: $[\epsilon_1, \epsilon_2, \epsilon_3]$.
* `t_substrate` (float): Coupling strength $t_s$ between each QD and the substrate (assumed to be the same for all QDs).
* `E_sub` (float, optional): Substrate energy level $E_{sub}$. Defaults to 0.0.
* `t0` (float, optional): Tip coupling strength prefactor $t_0$. Defaults to 0.1.
* `eta` (float, optional): Infinitesimal broadening parameter $\eta$. Defaults to 0.01.
* `n_qds` (int): Number of quantum dots (fixed at 3 in this implementation).
* `H_sub_QD` (np.ndarray): 1D array representing the coupling between substrate and QDs: $[t_s, t_s, t_s]$.
* `K_matrix` (np.ndarray): 3x3 matrix representing inter-dot Coulomb interactions:
    ```
    K_matrix = [[0, K, K],
                [K, 0, K],
                [K, K, 0]]
    ```
* `H_QD_no_tip` (np.ndarray): 4x4 matrix representing the Hamiltonian of the substrate + QDs system *without* the tip:
    ```
    H_QD_no_tip =  [[E_sub, t_s, t_s, t_s],
                   [t_s, eps_1, K, K],
                   [t_s, K, eps_2, K],
                   [t_s, K, K, eps_3]]
    ```

## Methods

### `calculate_greens_function(E, H_QD)`

Calculates the retarded Green's function $G^r(E)$:

$G^r(E) = [(E + i\eta)I - H_{QD}]^{-1}$

where $I$ is the 4x4 identity matrix.

**Arguments:**

* `E` (float): Energy $E$ at which to calculate the Green's function.
* `H_QD` (np.ndarray): 4x4 Hamiltonian matrix $H_{QD}$ (including the tip coupling).

**Returns:**

* `Gr` (np.ndarray): 4x4 complex-valued retarded Green's function matrix $G^r(E)$.

### `calculate_gamma(coupling_vector)`

Calculates the broadening matrix $\Gamma$ (in the wide-band limit):

$\Gamma = 2\pi V V^\dagger$

where $V$ is the coupling vector.

**Arguments:**

* `coupling_vector` (np.ndarray): Coupling vector $V$ (either $V_t$ for the tip or $W_s$ for the substrate).

**Returns:**

* `Gamma` (np.ndarray): 4x4 broadening matrix $\Gamma$.

### `calculate_transmission(tip_position, E)`

Calculates the transmission probability $T(E)$ for a given tip position and energy.

**Arguments:**

* `tip_position` (np.ndarray): 1D array representing the STM tip position $[x_{tip}, y_{tip}, z_{tip}]$.
* `E` (float): Energy $E$ at which to calculate the transmission.

**Returns:**

* `T` (float): Transmission probability $T(E)$.

**Internal Calculation Steps:**

1. **Constructs the tip coupling vector** $V_t$:
   $V_{ti} = t_0 e^{-\beta |r_{tip} - r_i|}$ for $i=1, 2, 3$ (for each QD). $V_{t0} = 0$ (no direct tip-substrate coupling).

2. **Constructs the full Hamiltonian** $H_{QD}$ (including tip coupling):
   $H_{QD} = H_{QD\_no\_tip} + \begin{bmatrix} 0 & V_t^\dagger \\ V_t & 0_{3x3} \end{bmatrix}$

3. **Calculates the Green's functions:** $G^r(E)$ and $G^a(E) = [G^r(E)]^\dagger$.

4. **Calculates the broadening matrices:** $\Gamma_t$ (for the tip) and $\Gamma_s$ (for the substrate). The substrate coupling vector $W_s$ is $[0, t_s, t_s, t_s]^T$.

5. **Calculates the transmission probability:**
   $T(E) = \text{Tr}[\Gamma_t G^r(E) \Gamma_s G^a(E)]$

## Example Usage

```python
# ... (Example usage code as in the previous response)
```

This documentation provides a comprehensive description of the `LandauerQDs` class, its attributes, and methods, using mathematical notation for clarity. This should help you understand and use the class effectively.
