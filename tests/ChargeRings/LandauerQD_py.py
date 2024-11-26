import numpy as np

import test_utils as tu

class LandauerQDs:
    def __init__(self, QDpos, Esite, K=0.01, decay=1.0, tS=0.01, E_sub=0.0, E_tip=0.0, tA=0.1, eta=0.00, Gamma_tip=1.0, Gamma_sub=1.0, debug=False ):
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

        self.n_qds = len(QDpos)  # Store the number of QDs

        # Construct the substrate-QD coupling matrix (constant for now)
        self.H_sub_QD = tS * np.ones(self.n_qds, dtype=np.complex128)

        # Construct the inter-QD Coulomb interaction matrix
        self.K_matrix = K * (np.ones((self.n_qds, self.n_qds)) - np.identity(self.n_qds))

        # Construct the bare QD Hamiltonian (without tip or substrate)
        self.Hqd0 = np.diag(self.Esite) + self.K_matrix
        
        if self.debug:
            tu.write_matrix( self.Hqd0, None,          "Hqd0 (LandauerQD_py.py)" )
            tu.write_matrix( self.Hqd0, "py_Hqd0.txt", "Hqd0 (LandauerQD_py.py)" )

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
        MIN_DIST      = 1e-3  # Minimum distance to prevent division by zero
        shifts = np.zeros(self.n_qds)
        for i in range(self.n_qds):
            d = tip_pos - self.QDpos[i]
            dist = np.linalg.norm(d)
            # if (dist < MIN_DIST){ printf("ERROR in calculate_tip_induced_shift(): dist(%g)<MIN_DIST(%g) tip_pos(%g,%g,%g) qd_pos(%g,%g,%g) \n", dist, MIN_DIST, tip_pos.x, tip_pos.y, tip_pos.z, qd_pos.x, qd_pos.y, qd_pos.z); exit(0); };
            if dist < MIN_DIST:
                print(f"ERROR in calculate_tip_induced_shift(): dist({dist})<MIN_DIST({MIN_DIST}) tip_pos({tip_pos}) qd_pos({self.QDpos[i]})"); 
                exit(0)
            #print( f"calculate_tip_induced_shift(): Q_tip({Q_tip}) dist({dist}) d({d}) qd_pos({self.QDpos[i]})" );
            shifts[i] = COULOMB_CONST * Q_tip / dist
        return shifts

    def _makeHqd(self, tip_pos, Q_tip=None):
        """
        Internal function to create QD Hamiltonian block with tip effects.
        
        Args:
            tip_pos: np.ndarray - Tip position
            Q_tip: float (optional) - Tip charge for energy shifts
            
        Returns:
            np.ndarray - QD Hamiltonian block including tip effects
        """
        
        # Start with the bare QD Hamiltonian
        Hqd = self.Hqd0.copy()
        
        # Add energy shifts from tip if Q_tip provided
        if Q_tip is not None:
            energy_shifts = self.calculate_tip_induced_shifts(tip_pos, Q_tip)
            if self.debug:
                print(f"LandauerQD_py._makeHqd() tip_pos={tip_pos} Q_tip={Q_tip} energy_shifts = {energy_shifts}")
            np.fill_diagonal(Hqd, np.diag(Hqd) + energy_shifts)  # Add shifts only to diagonal elements
            
        return Hqd

    def _assemble_full_H(self, tip_pos, Hqd, V_bias=0.0):
        """
        Internal function to assemble full Hamiltonian matrix.
        
        Args:
            tip_pos: np.ndarray - Tip position
            Hqd:     np.ndarray - Central QD Hamiltonian block
            V_bias:  float - Bias voltage (optional)
            
        Returns:
            np.ndarray - Full system Hamiltonian
        """
        
        tip_couplings = self.calculate_tip_coupling(tip_pos)    # Calculate tip coupling
        H = np.zeros((self.n_qds + 2, self.n_qds + 2), dtype=np.complex128)      # Construct full Hamiltonian

        if self.debug:
            print( "LandauerQD_py._assemble_full_H() tip_couplings:\n", tip_couplings )
        
        # Fill QD block (with small broadening)
        H[1:self.n_qds+1, 1:self.n_qds+1] = Hqd
        #H[1:self.n_qds+1, 1:self.n_qds+1] -= 1j * self.eta * np.eye(self.n_qds)    # NOTE: this is done in calculate_greens_function(), don't do it here !!!

        # Fill substrate part (with broadening and bias shift)
        H[0,0]              = self.E_sub - V_bias/2.0 - 1j * self.Gamma_sub
        H[0,1:self.n_qds+1] = self.H_sub_QD
        H[1:self.n_qds+1,0] = self.H_sub_QD
                
        # Fill tip part (with broadening and bias shift)
        H[-1,-1]             = self.E_tip + V_bias/2.0 - 1j * self.Gamma_tip
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
        
        G                = self.calculate_greens_function(energy, H)
        Gdag             = G.conj().T;  
        Gamma_s, Gamma_t = self._calculate_coupling_matrices()
        
        if self.debug:

            tu.write_matrix(H,        None,              "H_full  (LandauerQD_py.py) ")

            tu.write_matrix(H,        "py_H.txt",        "H_full  (LandauerQD_py.py) ")
            tu.write_matrix(G,        "py_G.txt",        "G       (LandauerQD_py.py) ")
            tu.write_matrix(Gdag,     "py_Gdag.txt",     "Gdag    (LandauerQD_py.py) ")
            tu.write_matrix(Gamma_s,  "py_Gamma_s.txt",  "Gamma_s (LandauerQD_py.py) ")
            tu.write_matrix(Gamma_t,  "py_Gamma_t.txt",  "Gamma_t (LandauerQD_py.py) ")

            Gdag          = G.conj().T;               
            Gammat_Gdag   = Gamma_t @ Gdag;           tu.write_matrix(Gammat_Gdag,   "py_Gammat_Gdag.txt",   "Gamma_t @ Gdag (LandauerQD_py.py)")
            G_Gammat_Gdag = G @ Gammat_Gdag;          tu.write_matrix(G_Gammat_Gdag, "py_G_Gammat_Gdag.txt", "G @ Gamma_t @ Gdag (LandauerQD_py.py)")
            Tmat          = Gamma_s @ G_Gammat_Gdag;  tu.write_matrix(Tmat,          "py_Tmat.txt",          "Tmat = Gamma_s @ G @ Gamma_t @ Gdag (LandauerQD_py.py)")
        else:
            Tmat = Gamma_s @ G @ Gamma_t @ Gdag
        
        transmission = np.real(np.trace(Tmat))
        
        if self.debug:
            print( "Tmat diag (channels) :\n" )
            for i in range(Tmat.shape[0]): print( f"Tmat[{i},{i}]: ", Tmat[i,i] )
            print(f"\nFinal transmission: {transmission} (LandauerQD_py.py)")
        
        return transmission

    def make_full_hamiltonian(self, tip_pos, Q_tip=None, Hqd=None, V_bias=0.0):
        """Construct full Hamiltonian including tip coupling and Coulomb shifts."""
        if Hqd is None:
            if Q_tip is None:
                raise ValueError("ERROR in make_full_hamiltonian(): Either Q_tip or Hqd must be provided.")
            Hqd = self._makeHqd(tip_pos, Q_tip)
        
        if self.debug:
            tu.write_matrix(Hqd, None, "Hqd  (LandauerQD_py.py) ")

        return self._assemble_full_H(tip_pos, Hqd, V_bias)

    def calculate_transmission(self, tip_pos, E, *, Q_tip=None, Hqd=None, V_bias=0.0):
        """Calculate transmission probability for given tip position and energy.
        
        Args:
            tip_pos: np.ndarray - Tip position vector
            E: float - Energy at which to calculate transmission
            Q_tip: float (optional) - Tip charge for energy shifts
            Hqd: np.ndarray (optional) - Pre-computed QD Hamiltonian
            V_bias: float - Bias voltage
        
        Returns:
            float - Transmission probability
        """
        H = self.make_full_hamiltonian(tip_pos, Q_tip=Q_tip, Hqd=Hqd, V_bias=V_bias)
        return self._calculate_transmission_from_H(H, E)

    def scan_1D(self, ps_line, energies, Qtips=None, H_QDs=None, V_bias=0.0):
        """
        Perform 1D scan along given line of positions.
        
        Args:
            ps_line: np.ndarray of shape (n_points, 3) - Line of tip positions
            energies: np.ndarray - Energies at which to calculate transmission
            Qtips: np.ndarray of shape (n_points,) - Tip charges for each position
            H_QDs: Optional pre-calculated QD Hamiltonians
            V_bias: Bias voltage
            
        Returns:
            np.ndarray - Transmission probabilities of shape (n_points, n_energies)
        """
        n_points      = len(ps_line)
        n_energies    = len(energies)
        transmissions = np.zeros((n_points, n_energies))
        
        for i, tip_pos in enumerate(ps_line):
            if H_QDs is not None:
                Hqd  = H_QDs[i]
                H = self.make_full_hamiltonian(tip_pos, Hqd=Hqd, V_bias=V_bias)
            else:
                H = self.make_full_hamiltonian(tip_pos, Q_tip=Qtips[i], V_bias=V_bias)
            
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

    def scan_eigenvalues(self, ps_line, Qtips):
        """Calculate QD eigenvalues along scanning line."""
        n_points    = len(ps_line)
        eigenvalues = np.zeros((n_points, self.n_qds))
        for i, tip_pos in enumerate(ps_line):
            eigenvalues[i] = self.get_QD_eigenvalues(tip_pos, Qtips[i])
        return eigenvalues

    def calculate_current(self, tip_pos, energies, V_bias, Q_tip=None, Hqd=None, T=300.0):
        """
        Calculate current by integrating transmission over energy window.
        
        Args:
            tip_pos: np.ndarray - Tip position
            energies: np.ndarray - Energies at which to calculate transmission
            V_bias: float - Bias voltage
            Q_tip: float (optional) - Tip charge
            Hqd: np.ndarray (optional) - Pre-computed QD Hamiltonian
            T: float - Temperature in Kelvin
            
        Returns:
            float - Current in atomic units
        """
        kB = 8.617333262e-5  # Boltzmann constant in eV/K
        # Calculate transmission for each energy with bias voltage
        transmissions = np.array([self.calculate_transmission(tip_pos=tip_pos, E=E, Q_tip=Q_tip, Hqd=Hqd, V_bias=V_bias) for E in energies])
        
        # Fermi functions for tip and substrate
        f_tip = 1.0 / (1.0 + np.exp((energies - V_bias/2.0) / (kB * T)))
        f_sub = 1.0 / (1.0 + np.exp((energies + V_bias/2.0) / (kB * T)))
        
        # Integrate using trapezoidal rule
        integrand = transmissions * (f_tip - f_sub)
        current = np.trapz(integrand, energies)
        
        return current

    def calculate_didv(self, tip_pos, energies, V_bias, dV=0.01, Q_tip=None, Hqd=None, T=300.0):
        """
        Calculate dI/dV using finite difference.
        
        Args:
            tip_pos: np.ndarray - Tip position
            energies: np.ndarray - Energies at which to calculate transmission
            V_bias: float - Bias voltage
            dV: float - Small voltage difference for finite difference
            Q_tip: float (optional) - Tip charge
            Hqd: np.ndarray (optional) - Pre-computed QD Hamiltonian
            T: float - Temperature in Kelvin
            
        Returns:
            float - Differential conductance (dI/dV)
        """
        # Calculate currents at V_bias ± dV/2
        I_plus = self.calculate_current(tip_pos, energies, V_bias + dV/2, Q_tip, Hqd, T)
        I_minus = self.calculate_current(tip_pos, energies, V_bias - dV/2, Q_tip, Hqd, T)
        
        # Calculate dI/dV using central difference
        didv = (I_plus - I_minus) / dV
        
        return didv

    def scan_didv_1D(self, ps_line, energies, V_bias, dV=0.01, Qtips=None, H_QDs=None, T=300.0):
        """
        Perform 1D scan of dI/dV along given line of positions.
        
        Args:
            ps_line: np.ndarray of shape (n_points, 3) - Line of tip positions
            energies: np.ndarray - Energies at which to calculate transmission
            V_bias: float - Bias voltage
            dV: float - Small voltage difference for finite difference
            Q_tip: float (optional) - Tip charge
            H_QDs: np.ndarray (optional) - Pre-computed QD Hamiltonians
            T: float - Temperature in Kelvin
            
        Returns:
            np.ndarray - dI/dV values along the scanning line
        """
        n_points = len(ps_line)
        didv_values = np.zeros(n_points)
        
        for i in range(n_points):
            Hqd = H_QDs[i] if H_QDs is not None else None
            didv_values[i] = self.calculate_didv(ps_line[i], energies, V_bias, dV, Qtips[i], Hqd, T)
            
        return didv_values

    def scan_didv_2D(self, ps_line, energies, V_bias, dV=0.01, Qtips=None, H_QDs=None, T=300.0):
        """
        Perform 2D scan of dI/dV along given line of positions and energies.
        
        Args:
            ps_line: np.ndarray of shape (n_points, 3) - Line of tip positions
            energies: np.ndarray - Energies at which to calculate transmission
            V_bias: float - Bias voltage
            dV: float - Small voltage difference for finite difference
            Qtips: np.ndarray (optional) - Array of tip charges for each position
            H_QDs: np.ndarray (optional) - Pre-computed QD Hamiltonians
            T: float - Temperature in Kelvin
            
        Returns:
            np.ndarray - dI/dV values of shape (n_points, n_energies)
        """
        n_points = len(ps_line)
        n_energies = len(energies)
        didv_map = np.zeros((n_points, n_energies))
        
        for i, tip_pos in enumerate(ps_line):
            Hqd = H_QDs[i] if H_QDs is not None else None
            Q_tip = Qtips[i] if Qtips is not None else None
            for j, E in enumerate(energies):
                # Calculate transmission at V_bias ± dV/2
                T_plus = self.calculate_transmission(tip_pos, E, Q_tip=Q_tip, Hqd=Hqd, V_bias=V_bias + dV/2)
                T_minus = self.calculate_transmission(tip_pos, E, Q_tip=Q_tip, Hqd=Hqd, V_bias=V_bias - dV/2)
                # Approximate dI/dV
                didv_map[i,j] = (T_plus - T_minus) / dV
                
        return didv_map

if __name__ == "__main__":
    QDpos = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0]])
    K = 0.01  # Coulomb interaction
    decay = 1.0  # Decay constant
    Esite = np.array([0.1, 0.2, 0.3])
    tS = 0.01  # QD-substrate coupling

    system = LandauerQDs(QDpos, Esite, K, decay, tS, debug=True)

    tip_position = np.array([0.0, 0.0, 1.0])
    E = 0.5
    transmission = system.calculate_transmission(tip_position, E)
    print(f"Transmission: {transmission}")