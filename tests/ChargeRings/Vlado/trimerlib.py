def SetHamiltonian(**kwargs):
	'''
	eps1, eps2, eps3 - local energy levels (meV)
	T1, T2, T3 - transmission of the coupling of the QDs to the tip (dimensionless)
	VS - coupling to the substrate
	eV - tip voltage
	muS, muT, Temp - chemical potential of the substrate/tip, temperature
	t - direct hopping (meV)
	W - intersite coupling (meV)
	U - local Coulomb interaction (meV) (only for spinless=False)
	J - exchange coupling of Heisenberg type

	'''
	eps1, eps2, eps3 = kwargs.get('eps1'), kwargs.get('eps2'), kwargs.get('eps3')
	VS = kwargs.get('VS')
	T1, T2, T3 = kwargs.get('T1'), kwargs.get('T2'), kwargs.get('T3')
	eV, muS, muT, Temp = kwargs.get('eV'), kwargs.get('muS'), kwargs.get('muT'), kwargs.get('Temp')
	U = kwargs.get('U',0.0)	
	t = kwargs.get('t',0.0)
	W = kwargs.get('W',0.0)
	J = kwargs.get('J',0.0)

	if kwargs.get('spinless'):
		## one-particle Hamiltonian
		H1p = {(0,0): eps1, (0,1): t,    (0,2): t,
	     	               (1,1): eps2, (1,2): t,
	     	                            (2,2): eps3 }
	
		## two-particle Hamiltonian: inter-site coupling
		H2p = {(0,1,1,0): W,
		       (1,2,2,1): W,
		       (0,2,2,0): W }
	
		## leads: substrate (S) and scanning tip (T)
		LeadMus   = {0: muS,  1: muT+eV }
		LeadTemps = {0: Temp, 1: Temp  }
	
		## coupling between leads (1st number) and impurities (2nd number)
		TLeads = {(0,0): VS,     # S <--> 1
		          (0,1): VS,     # S <--> 2
		          (0,2): VS,     # S <--> 3
		          (1,0): VS*T1,  # T <--> 1
		          (1,1): VS*T2,  # T <--> 2
		          (1,2): VS*T3 } # T <--> 3
	else:	
		## one-particle Hamiltonian
	     ## 0: 1up, 1: 1dn, 2: 2up, 3: 2dn, 4: 3up, 5: 3dn
		H1p = {(0,0): eps1, (0,1): 0.0,  (0,2): t,    (0,3): 0.0,  (0,4): t,    (0,5): 0.0,
	     	               (1,1): eps1, (1,2): 0.0,  (1,3): t,    (1,4): 0.0,  (1,5): t,
	     	                            (2,2): eps2, (2,3): 0.0,  (2,4): t,    (2,5): 0.0,
	     	                                         (3,3): eps2, (3,4): 0.0,  (3,5): t,
	     	                                                      (4,4): eps3, (4,5): 0.0,
	     	                                                                   (5,5): eps3 }
	
		## two-particle Hamiltonian: on-site and inter-site coupling
		H2p = {(0,1,1,0): U, (2,3,3,2): U, (4,5,5,4): U,
		       (0,2,2,0): W + J/4.0, (0,3,3,0): W - J/4.0, (0,4,4,0): W + J/4.0, (0,5,5,0): W - J/4.0,
		       (1,2,2,1): W - J/4.0, (1,3,3,1): W + J/4.0, (1,4,4,1): W - J/4.0, (1,5,5,1): W + J/4.0,
		       (2,4,4,2): W + J/4.0, (2,5,5,2): W - J/4.0,
		       (3,4,4,3): W - J/4.0, (3,5,5,3): W + J/4.0,
		       (0,3,1,2): -J/2.0, (1,2,0,3): -J/2.0,
		       (0,5,1,4): -J/2.0, (1,4,0,5): -J/2.0,
		       (2,5,3,4): -J/2.0, (3,4,2,5): -J/2.0 }

		## leads: substrate (S) and scanning tip (T), no spin-splitting
		## 0: Sup, 1: Sdn, 2: Tup, 3: Tdn
		LeadMus   = {0: muS,  1: muS,  2: muT+eV,  3: muT+eV }
		LeadTemps = {0: Temp, 1: Temp, 2: Temp,    3: Temp   }
	
		## coupling between leads (1st number) and impurities (2nd number)
		TLeads = {(0,0): VS, # Sup <--> 1up
		          (1,1): VS, # Sdn <--> 1dn
		          (0,2): VS, # Sup <--> 2up
		          (1,3): VS, # Sdn <--> 2dn
		          (0,4): VS, # Sup <--> 3up
		          (1,5): VS, # Sdn <--> 3dn
				(2,0): VS*T1, # Tup <--> 1up
		          (3,1): VS*T1, # Tdn <--> 1dn
		          (2,2): VS*T2, # Tup <--> 2up
		          (3,3): VS*T2, # Tdn <--> 2dn
		          (2,4): VS*T3, # Tup <--> 3up
		          (3,5): VS*T3} # Tdn <--> 3dn

	return H1p,H2p,LeadMus,LeadTemps,TLeads

## trimerlib END

