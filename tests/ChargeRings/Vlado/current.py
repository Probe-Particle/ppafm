import numpy as np

import sys
sys.path.insert(0, '/home/prokop/git_SW/qmeq/') ## python version
import qmeq
from qmeq import config as qmeq_config
#from trimerlib import SetHamiltonian

from configparser import ConfigParser

default_params = {
    'U'    : 0.0,
    'W'    : 0.0,
    't'    : 0.0,
    'J'    : 0.0,
    'muS'  : 0.0,
    'muT'  : 0.0,
    'VS'   : 0.0,
    'VT'   : 0.0,
    'GammaS' : 0.0,
    'GammaT' : 0.0,
    'VBias': 0.0,
    'Temp' : 0.0,
    'DBand': 100.0,
    'NSingle' : 3,
    'NLeads'  : 2,
    'spinless': True,
    'solver'  : 'Pauli',
    
}

def load_parameters(config_file='qmeq.in', params=default_params ):
    #config = ConfigParser()
    config = ConfigParser(comment_prefixes=(';',))
    config.read('qmeq.in')
    try:
        config.read(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return None  # Indicate failure
    try:
        for key in params:
            if config.has_option('params', key):
                value = config.get('params', key)
                if isinstance(params[key], bool):
                    params[key] = bool(int(value))
                elif isinstance(params[key], int):
                    params[key] = int(value)
                elif isinstance(params[key], float):
                    params[key] = float(value)
                else:  # string
                    params[key] = value
    except (KeyError, ValueError) as e:
        print(f"Error reading configuration: {e}")
        return None  # Indicate failure
    return params


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

def SetHamiltonian_spinless(**kwargs):
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
    t = kwargs.get('t',0.0)
    W = kwargs.get('W',0.0)

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
    
    return H1p,H2p,LeadMus,LeadTemps,TLeads


def calculate_current(params, input_data, verbosity=0):
    """
    Calculate current for given parameters and input data
    """
    positions = input_data[:, 0]
    eps1 = input_data[:, 3] * 1000.0
    eps2 = input_data[:, 4] * 1000.0
    eps3 = input_data[:, 5] * 1000.0
    T1   = input_data[:, 9]
    T2   = input_data[:, 10]
    T3   = input_data[:, 11]

    currents = np.zeros(len(positions))

    VBias = params['VBias']
    U     = params['U']
    W     = params['W']
    t     = params['t']
    J     = params['J']
    muS   = params['muS']
    muT   = params['muT']
    VS    = params['VS']
    VT    = params['VT']
    Temp = params['Temp']
    spinless = params['spinless']

    for i in range(len(positions)):

        if verbosity > 0:
            print("\n#####################\n");
            print("### QmeQ current.py : calculate_current()  # i: %i \n", i);
            print("#####################\n\n");
        #print( f"calculate_current() i: {i}   VBias: {VBias}  eps: {eps1[i]} {eps2[i]} {eps3[i]}")

        #H1p, H2p, LeadMus, LeadTemps, TLeads = SetHamiltonian(
        H1p, H2p, LeadMus, LeadTemps, TLeads = SetHamiltonian_spinless(
            eV=VBias, eps1=eps1[i], eps2=eps2[i], eps3=eps3[i],
            T1=T1[i], T2=T2[i], T3=T3[i], U=U, W=W, t=t, J=J,
            muS=muS, muT=muT, VS=VS, VT=VT, Temp=Temp
        )

        system = qmeq.Builder(
            params['NSingle'], H1p, H2p,
            params['NLeads'], TLeads, LeadMus, LeadTemps,
            params['DBand'], kerntype=params['solver'],
            indexing='Lin', itype=0, symq=1,
            solmethod='lsqr', mfreeq=0, norm_row=0
        )

        qmeq_config.verbosity = verbosity
        system.appr.verbosity = verbosity  # Set verbosity after instance creation
        system.verbosity = verbosity
        system.solve()

        if params['spinless']:
            currents[i] = system.current[1]
        else:
            currents[i] = system.current[2] + system.current[3]

    return positions, currents

def _current_worker(params, chunk):
    """
    Worker function for parallel current calculation (must be at module level)
    """
    positions = chunk[:, 0]
    eps1 = chunk[:, 3] * 1000.0
    eps2 = chunk[:, 4] * 1000.0
    eps3 = chunk[:, 5] * 1000.0
    T1   = chunk[:, 9]
    T2   = chunk[:, 10]
    T3   = chunk[:, 11]
    
    currents = np.zeros(len(positions))
    
    for i in range(len(positions)):
        H1p, H2p, LeadMus, LeadTemps, TLeads = SetHamiltonian(
            eV=params['VBias'], eps1=eps1[i], eps2=eps2[i], eps3=eps3[i],
            T1=T1[i], T2=T2[i], T3=T3[i], U=params['U'], W=params['W'], 
            t=params['t'], J=params['J'], muS=params['muS'], muT=params['muT'],
            VS=params['VS'], VT=params['VT'], Temp=params['Temp'], 
            spinless=params['spinless']
        )
    
        system = qmeq.Builder(
            params['NSingle'], H1p, H2p,
            params['NLeads'], TLeads, LeadMus, LeadTemps,
            params['DBand'], kerntype=params['solver'],
            indexing='Lin', itype=0, symq=1,
            solmethod='lsqr', mfreeq=0, norm_row=0,
            verbosity = verbosity

        )
        
        system.solve()
        
        if params['spinless']:
            currents[i] = system.current[1]
        else:
            currents[i] = system.current[2] + system.current[3]
    
    return positions, currents

def calculate_current_parallel(params, input_data, nThreads=4):
    """
    Parallel version of calculate_current() using multiprocessing.Pool
    """
    import multiprocessing as mp
    import numpy as np
    
    # Split input data into chunks for each thread
    chunk_size = len(input_data) // nThreads
    chunks = [input_data[i*chunk_size:(i+1)*chunk_size] for i in range(nThreads-1)]
    chunks.append(input_data[(nThreads-1)*chunk_size:])
    
    # Create partial function with params
    from functools import partial
    worker = partial(_current_worker, params)
    
    # Process chunks in parallel
    with mp.Pool(processes=nThreads) as pool:
        results = pool.map(worker, chunks)
    
    # Combine results
    positions = np.concatenate([r[0] for r in results])
    currents = np.concatenate([r[1] for r in results])
    
    return positions, currents

def calculate_didv(positions, bias_voltages, current_grid):
    """
    Calculate differential conductance
    """
    didv_grid = np.zeros_like(current_grid)
    print( f"shapes: bias_voltages {bias_voltages.shape}, current_grid {current_grid.shape}" )
    for i in range(len(positions)):
        didv_grid[:, i] = np.gradient(current_grid[:, i], bias_voltages)
    return didv_grid
