import numpy as np

class LandauerQDs:
    def __init__(self, QDpos, Esite, K=0.01, decay=1.0, tS=0.01, E_sub=0.0, E_tip=0.0, tA=0.1, eta=0.00, Gamma_tip=1.0, Gamma_sub=1.0):
        """
        Initializes the LandauerQDs class.

        Args:
            QDpos     : np.ndarray : A 3x3 array of QD positions.
            K        : float      : Coulomb interaction between QDs (assumed same for all).
            decay    : float      : Decay constant for tip-QD coupling.
            Esite    : np.ndarray : A 1D array of on-site energies for the QDs.
            tS       : float      : Coupling strength between QDs and substrate (same for all).
            E_sub    : float      : Substrate energy level. Defaults to 0.0.
            tA       : float      : Tip coupling strength prefactor. Defaults to 0.1.
            eta      : float      : Infinitesimal broadening parameter for QDs. Defaults to 0.01.
            Gamma_tip: float      : Broadening of the tip state. Defaults to 1.0.
            Gamma_sub: float      : Broadening of the substrate state. Defaults to 1.0.
        """
        self.QDpos = QDpos
        self.K = K
        self.decay = decay
        self.Esite = Esite
        self.tS = tS
        self.E_sub = E_sub
        self.E_tip = E_tip
        self.tA = tA
        self.eta = eta
        self.Gamma_tip = Gamma_tip
        self.Gamma_sub = Gamma_sub

        self.n_qds = len(QDpos)  # Store the number of QDs

        # Construct the substrate-QD coupling matrix (constant for now)
        self.H_sub_QD = tS * np.ones(self.n_qds, dtype=np.complex128)

        # Construct the inter-QD Coulomb interaction matrix
        self.K_matrix = K * (np.ones((self.n_qds, self.n_qds)) - np.identity(self.n_qds))

        # Construct the main block of the Hamiltonian (H_QD) - without tip yet
        self.H_QD_no_tip = np.block([
            [np.array([[self.E_sub]]), self.H_sub_QD[None, :]],
            [self.H_sub_QD[:, None], np.diag(self.Esite) + self.K_matrix]
        ])

    def calculate_greens_function(self, E, H_QD):
        """Calculates the retarded Green's function."""
        identity = np.identity(len(H_QD), dtype=np.complex128)
        return np.linalg.inv((E + 1j*self.eta)*identity - H_QD)

    def calculate_gamma(self, coupling_vector):
        """Calculates the broadening matrix (wide-band limit)."""
        return 2 * np.pi * np.outer(coupling_vector, np.conj(coupling_vector))

    def calculate_tip_coupling(self, tip_pos):
        """Calculate coupling between tip and QDs based on distance."""
        tip_couplings = np.zeros(self.n_qds, dtype=np.complex128)
        for i in range(self.n_qds):
            d = np.linalg.norm(tip_pos - self.QDpos[i])
            tip_couplings[i] = self.tA * np.exp(-self.decay * d)
        return tip_couplings

    def calculate_tip_induced_shifts(self, tip_pos, Q_tip):
        """Calculate energy shifts induced by tip's Coulomb potential."""
        COULOMB_CONST = 14.3996  # eV*Å/e
        shifts = np.zeros(self.n_qds)
        for i in range(self.n_qds):
            d = tip_pos - self.QDpos[i]
            shifts[i] = COULOMB_CONST * Q_tip / np.linalg.norm(d)
        return shifts

    def make_full_hamiltonian(self, tip_pos, Q_tip):
        """Construct full Hamiltonian including tip coupling and Coulomb shifts."""
        # Calculate tip coupling and energy shifts
        tip_couplings = self.calculate_tip_coupling(tip_pos)
        energy_shifts = self.calculate_tip_induced_shifts(tip_pos, Q_tip)
        
        # Update QD energies with tip-induced shifts
        shifted_energies = self.Esite + energy_shifts
        
        # Construct QD block with shifted energies
        H_QD_block = np.diag(shifted_energies) + self.K_matrix
        
        # Construct full Hamiltonian with complex energies for tip and substrate
        H = np.zeros((self.n_qds + 2, self.n_qds + 2), dtype=np.complex128)
        
        # Fill QD block (with small broadening)
        H[1:self.n_qds+1,1:self.n_qds+1] = H_QD_block - 1j * self.eta * np.eye(self.n_qds)

        # Fill substrate part (with broadening)
        H[0,0] = self.E_sub - 1j * self.Gamma_sub     # Substrate on-site
        H[0,1:self.n_qds+1] = self.H_sub_QD           # Substrate-QD coupling
        H[1:self.n_qds+1,0] = self.H_sub_QD           # QD-substrate coupling
                
        # Fill tip part (with broadening)
        H[-1,-1]             = self.E_tip - 1j * self.Gamma_tip  # Tip on-site
        H[1:self.n_qds+1,-1] = tip_couplings                     # QD-tip coupling 
        H[-1,1:self.n_qds+1] = tip_couplings.conj()              # Tip-QD coupling
        
        return H

    def get_QD_eigenvalues(self, tip_pos, Q_tip):
        """Calculate eigenvalues of the QD subsystem for given tip position."""
        # Get energy shifts from tip
        energy_shifts = self.calculate_tip_induced_shifts(tip_pos, Q_tip)
        
        # Construct QD Hamiltonian (without tip and substrate)
        H_QD = np.diag(self.Esite + energy_shifts) + self.K_matrix
        
        # Calculate eigenvalues
        return np.linalg.eigvalsh(H_QD)

    def scan_eigenvalues(self, ps_line, Q_tip):
        """Calculate QD eigenvalues along scanning line."""
        n_points = len(ps_line)
        eigenvalues = np.zeros((n_points, self.n_qds))
        
        for i, tip_pos in enumerate(ps_line):
            eigenvalues[i] = self.get_QD_eigenvalues(tip_pos, Q_tip)
            
        return eigenvalues

    def calculate_transmission(self, tip_pos, E, Q_tip=0.0 ):
        """Calculates the transmission probability for a given tip position and energy."""
        # Get the full Hamiltonian including tip effects
        H = self.make_full_hamiltonian(tip_pos, Q_tip=Q_tip )  # Q_tip=0 for transmission calc
        
        # Calculate Green's function
        G = self.calculate_greens_function(E, H)
        
        # Calculate coupling matrices (wide-band limit)
        Gamma_tip = np.zeros_like(H)
        Gamma_tip[-1,-1] = 2 * np.pi  # Tip coupling
        
        Gamma_sub = np.zeros_like(H)
        Gamma_sub[0,0] = 2 * np.pi  # Substrate coupling
        
        # Calculate transmission
        temp = Gamma_tip @ G @ Gamma_sub @ G.conj().T
        return np.real(np.trace(temp))

    def scan_1D(self, ps_line, Q_tip, energies):
        """
        Perform 1D scan along given line of positions.
        
        Args:
            ps_line: np.ndarray of shape (n_points, 3) - Line of tip positions
            Q_tip: float - Tip charge
            energies: np.ndarray - Energies at which to calculate transmission
            
        Returns:
            transmissions: np.ndarray of shape (n_points, n_energies)
        """
        n_points = len(ps_line)
        n_energies = len(energies)
        transmissions = np.zeros((n_points, n_energies))
        
        for i, tip_pos in enumerate(ps_line):
            H = self.make_full_hamiltonian(tip_pos, Q_tip)
            
            for j, E in enumerate(energies):
                # Calculate Green's function
                G = self.calculate_greens_function(E, H)
                
                # Calculate coupling matrices (wide-band limit)
                Gamma_tip = np.zeros_like(H)
                Gamma_tip[-1,-1] = 2 * np.pi  # Tip coupling
                
                Gamma_sub = np.zeros_like(H)
                Gamma_sub[0,0] = 2 * np.pi  # Substrate coupling
                
                # Calculate transmission
                temp = Gamma_tip @ G @ Gamma_sub @ G.conj().T
                transmissions[i,j] = np.real(np.trace(temp))
        
        return transmissions

if __name__ == "__main__":
    QDpos = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0]])
    K = 0.01  # Coulomb interaction
    decay = 1.0  # Decay constant
    Esite = np.array([0.1, 0.2, 0.3])
    tS = 0.01  # QD-substrate coupling

    system = LandauerQDs(QDpos, Esite, K, decay, tS)

    tip_position = np.array([0.0, 0.0, 1.0])
    E = 0.5
    transmission = system.calculate_transmission(tip_position, E)
    print(f"Transmission: {transmission}")