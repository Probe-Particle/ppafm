import numpy as np

import sys
sys.path.insert(0, '/home/prokop/git_SW/qmeq/') ## python version
import qmeq
from trimerlib import SetHamiltonian

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



def calculate_current(params, input_data):
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
        #print( f"calculate_current() i: {i}   VBias: {VBias}  eps: {eps1[i]} {eps2[i]} {eps3[i]}")

        H1p, H2p, LeadMus, LeadTemps, TLeads = SetHamiltonian(
            eV=VBias, eps1=eps1[i], eps2=eps2[i], eps3=eps3[i],
            T1=T1[i], T2=T2[i], T3=T3[i], U=U, W=W, t=t, J=J,
            muS=muS, muT=muT, VS=VS, VT=VT, Temp=Temp, spinless=spinless
        )

        system = qmeq.Builder(
            params['NSingle'], H1p, H2p,
            params['NLeads'], TLeads, LeadMus, LeadTemps,
            params['DBand'], kerntype=params['solver'],
            indexing='Lin', itype=0, symq=1,
            solmethod='lsqr', mfreeq=0, norm_row=0
        )
        system.solve()

        if params['spinless']:
            currents[i] = system.current[1]
        else:
            currents[i] = system.current[2] + system.current[3]

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
