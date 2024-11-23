import numpy as np

import test_utils as tu

class LandauerQDs:
    def __init__(self, QDpos, Esite, K=0.01, decay=1.0, tS=0.01, E_sub=0.0, E_tip=0.0, tA=0.1, eta=0.00, Gamma_tip=1.0, Gamma_sub=1.0, debug=False, verbosity=0):
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
            debug    : bool      : Enable debug output. Defaults to False.
            verbosity: int       : Level of verbosity (0-2). Defaults to 0.
        """
        self.QDpos = QDpos
        self.K = K
        self.decay = decay
        self.Esite = Esite
        self.tS    = tS
        self.E_sub = E_sub
        self.E_tip = E_tip
        self.tA    = tA
        self.eta   = eta
        self.Gamma_tip = Gamma_tip
        self.Gamma_sub = Gamma_sub
        self.debug = debug
        self.verbosity = verbosity

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

    def calculate_greens_function(self, E, H):
        """
        Calculates the retarded Green's function.
        
        Args:
            E: float - Energy at which to calculate Green's function
            H: np.ndarray - full Hamiltonian matrix
            
        Returns:
            np.ndarray - Retarded Green's function
        """
        identity = np.eye(len(H), dtype=np.complex128)
        G = np.linalg.inv((E + 1j*self.eta)*identity - H)
        
        if self.debug and self.verbosity > 1:
            print("\nPython Green's function calculation:")
            print("Original matrix ((E + iη)I - H):")
            A = (E + 1j*self.eta)*identity - H
            print(A)
            
            print("\nInverted matrix G:")
            print(G)
            
            # Verify inversion
            verification = np.matmul(A, G)
            print("\nVerification A * A^-1:")
            print(verification)
            
            # Check deviation from identity
            off_diag_mask = ~np.eye(len(H), dtype=bool)
            max_off_diag  = np.max(np.abs(verification[off_diag_mask]))
            max_diag_diff = np.max(np.abs(verification[np.eye(len(H), dtype=bool)] - 1))
            print(f"Max deviation from identity: diagonal={max_diag_diff:.2e} off-diagonal={max_off_diag:.2e}")
        
        return G

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

    def _makeHqd(self, tip_pos, Q_tip=None):
        """
        Internal function to create QD Hamiltonian block.
        
        Args:
            tip_pos: np.ndarray - Tip position
            Q_tip: float (optional) - Tip charge for energy shifts
            
        Returns:
            np.ndarray - QD Hamiltonian block
        """
            
        # Calculate energy shifts from tip if Q_tip provided
        if Q_tip is not None:
            energy_shifts    = self.calculate_tip_induced_shifts(tip_pos, Q_tip)
            shifted_energies = self.Esite + energy_shifts
        else:
            shifted_energies = self.Esite
            
        # Construct QD block with shifted energies
        Hqd = np.diag(shifted_energies) + self.K_matrix
        return Hqd

    def _assemble_full_H(self, tip_pos, Hqd ):
        """
        Internal function to assemble full Hamiltonian matrix.
        
        Args:
            tip_pos: np.ndarray - Tip position
            Hqd:     np.ndarray - Central QD Hamiltonian block
            
        Returns:
            np.ndarray - Full system Hamiltonian
        """
        
        tip_couplings = self.calculate_tip_coupling(tip_pos)    # Calculate tip coupling
        H = np.zeros((self.n_qds + 2, self.n_qds + 2), dtype=np.complex128)      # Construct full Hamiltonian
        
        # Fill QD block (with small broadening)
        H[1:self.n_qds+1, 1:self.n_qds+1] = Hqd
        H[1:self.n_qds+1, 1:self.n_qds+1] -= 1j * self.eta * np.eye(self.n_qds)

        # Fill substrate part (with broadening)
        H[0,0]              = self.E_sub - 1j * self.Gamma_sub
        H[0,1:self.n_qds+1] = self.H_sub_QD
        H[1:self.n_qds+1,0] = self.H_sub_QD
                
        # Fill tip part (with broadening)
        H[-1,-1]             = self.E_tip - 1j * self.Gamma_tip
        H[1:self.n_qds+1,-1] = tip_couplings
        H[-1,1:self.n_qds+1] = tip_couplings.conj()
        
        return H

    def _calculate_coupling_matrices(self):
        """
        Internal function to build coupling matrices Gamma for tip and substrate.
        
        Returns:
            tuple(np.ndarray, np.ndarray) - (Gamma_substrate, Gamma_tip)
        """
        size = self.n_qds + 2
        Gamma_s = np.zeros((size, size), dtype=np.complex128);  Gamma_s[0,0] = 2 * self.Gamma_sub       # Substrate coupling
        Gamma_t = np.zeros((size, size), dtype=np.complex128);  Gamma_t[-1,-1] = 2 * self.Gamma_tip     # Tip coupling
        return Gamma_s, Gamma_t

    def _calculate_transmission_from_H(self, H, energy):
        """
        Internal function to calculate transmission from Hamiltonian using Caroli formula:  T = Tr( Gamma_s @ G @ Gamma_t @ G†  )

        
        Args:
            H: np.ndarray - Full system Hamiltonian
            energy: float - Energy at which to calculate transmission
            
        Returns:
            float - Transmission probability
        """
        if self.debug and self.verbosity > 0:
            print("\nPython transmission calculation:")
            print("Full Hamiltonian H:")
            print(H)
        
        tu.save_matrix(H,  "py_H.txt",  "H_full (LandauerQD_py.py) ")

        G = self.calculate_greens_function(energy, H)
        Gdag          = G.conj().T;  
        Gamma_s, Gamma_t = self._calculate_coupling_matrices()

        tu.save_matrix(H,        "py_H.txt",        "H_full  (LandauerQD_py.py) ")
        tu.save_matrix(G,        "py_G.txt",        "G       (LandauerQD_py.py) ")
        tu.save_matrix(Gdag,     "py_Gdag.txt",     "Gdag    (LandauerQD_py.py) ")
        tu.save_matrix(Gamma_s,  "py_Gamma_s.txt",  "Gamma_s (LandauerQD_py.py) ")
        tu.save_matrix(Gamma_t,  "py_Gamma_t.txt",  "Gamma_t (LandauerQD_py.py) ")
        
        if self.debug and self.verbosity > 1:
            print("\nGamma_substrate:")
            print(Gamma_s)
            print("\nGamma_tip:")
            print(Gamma_t)  
            Gdag          = G.conj().T;               
            Gammat_Gdag   = Gamma_t @ Gdag;           tu.save_matrix(Gammat_Gdag,   "py_Gammat_Gdag.txt",   "Gamma_t @ Gdag (LandauerQD_py.py)")
            G_Gammat_Gdag = G @ Gammat_Gdag;          tu.save_matrix(G_Gammat_Gdag, "py_G_Gammat_Gdag.txt", "G @ Gamma_t @ Gdag (LandauerQD_py.py)")
            Tmat          = Gamma_s @ G_Gammat_Gdag;  tu.save_matrix(Tmat,          "py_Tmat.txt",          "Tmat = Gamma_s @ G @ Gamma_t @ Gdag (LandauerQD_py.py)")
        else:
            Tmat = Gamma_s @ G @ Gamma_t @ Gdag
        
        transmission = np.real(np.trace(Tmat))
        
        if self.debug and self.verbosity > 0:
            print( "Tmat diag (channels) :\n" )
            for i in range(Tmat.shape[0]): print( f"Tmat[{i},{i}]: ", Tmat[i,i] )
            print(f"\nFinal transmission: {transmission} (LandauerQD_py.py)")
        
        return transmission

    def make_full_hamiltonian(self, tip_pos, Q_tip=None, Hqd=None):
        """Construct full Hamiltonian including tip coupling and Coulomb shifts."""
        if Hqd is None:
            if Q_tip is None:
                raise ValueError("ERROR in make_full_hamiltonian(): Either Q_tip or Hqd must be provided.")
            Hqd = self._makeHqd(tip_pos, Q_tip)
        return self._assemble_full_H(tip_pos, Hqd)

    def calculate_transmission(self, tip_pos, E, Q_tip=None, Hqd=None ):
        """Calculate transmission probability for given tip position and energy."""
        H = self.make_full_hamiltonian(tip_pos, Q_tip=Q_tip, Hqd=Hqd )
        return self._calculate_transmission_from_H(H, E)

    def scan_1D(self, ps_line, energies, Q_tip=None, H_QDs=None):
        """
        Perform 1D scan along given line of positions.
        
        Args:
            ps_line: np.ndarray of shape (n_points, 3) - Line of tip positions
            energies: np.ndarray - Energies at which to calculate transmission
            Q_tip: float (optional) - Tip charge
            H_QDs: np.ndarray (optional) - Pre-computed QD Hamiltonians
            
        Returns:
            np.ndarray - Transmission probabilities of shape (n_points, n_energies)
        """
        n_points      = len(ps_line)
        n_energies    = len(energies)
        transmissions = np.zeros((n_points, n_energies))
        
        for i, tip_pos in enumerate(ps_line):
            if H_QDs is not None:
                Hqd  = H_QDs[i]
                H = self.make_full_hamiltonian(tip_pos, Hqd=Hqd)
            else:
                H = self.make_full_hamiltonian(tip_pos, Q_tip=Q_tip)
            
            for j, E in enumerate(energies):
                transmissions[i,j] = self._calculate_transmission_from_H(H, E)
                
        return transmissions

    def get_QD_eigenvalues(self, tip_pos, Q_tip):
        """Calculate eigenvalues of the QD subsystem for given tip position."""
        #energy_shifts = self.calculate_tip_induced_shifts(tip_pos, Q_tip) # Get energy shifts from tip
        #H_QD = np.diag(self.Esite + energy_shifts) + self.K_matrix   # Construct QD Hamiltonian (without tip and substrate)
        Hqd = self._makeHqd( tip_pos, Q_tip=Q_tip)
        eigvals = np.linalg.eigvalsh(Hqd) # Calculate eigenvalues
        return eigvals

    def scan_eigenvalues(self, ps_line, Q_tip):
        """Calculate QD eigenvalues along scanning line."""
        n_points    = len(ps_line)
        eigenvalues = np.zeros((n_points, self.n_qds))
        for i, tip_pos in enumerate(ps_line):
            eigenvalues[i] = self.get_QD_eigenvalues(tip_pos, Q_tip)
        return eigenvalues

if __name__ == "__main__":
    QDpos = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0]])
    K = 0.01  # Coulomb interaction
    decay = 1.0  # Decay constant
    Esite = np.array([0.1, 0.2, 0.3])
    tS = 0.01  # QD-substrate coupling

    system = LandauerQDs(QDpos, Esite, K, decay, tS, debug=True, verbosity=2)

    tip_position = np.array([0.0, 0.0, 1.0])
    E = 0.5
    transmission = system.calculate_transmission(tip_position, E)
    print(f"Transmission: {transmission}")