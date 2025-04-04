import numpy as np
import matplotlib.pyplot as plt

import glob
import argparse
from current import calculate_current, calculate_didv, load_parameters, calculate_current_parallel
from plot_utils import plot_results, plot_results_1d

# Add C++ solver imports
import sys
from sys import path
path.insert(0, '../../../pyProbeParticle')
import pauli as psl
from pauli import PauliSolver

params = load_parameters( config_file='qmeq.in' )

# print( "#### params: " )
# for k, v in params.items(): print(f"{k}: {v}")

params['VS'] = np.sqrt(params['GammaS']/np.pi)
params['VT'] = np.sqrt(params['GammaT']/np.pi)

input_files = glob.glob('input/*.dat')
input_files = sorted(input_files)

#print("#### input_files: ", input_files)

# input_files = [
#     'input/0.50_line_scan.dat',
#     'input/0.51_line_scan.dat',
#     'input/0.53_line_scan.dat',
# ]

print("#### params: ")
for k, v in params.items(): print(f"{k}: {v}")

def eval_dir_of_lines( input_files, params, Vmin = 0.0, Vmax = 10.0, nThreads=12  ):
    # Initialize arrays
    bias_voltages = []
    positions     = None
    eps_max_grid  = []
    current_grid  = []

    for input_file in input_files:
        # Extract bias voltage from filename
        filename = input_file.split('/')[-1]
        if not '_line_scan.dat' in filename: continue
        Vbias = float(filename.split('_line_scan.dat')[0])
        input_data = np.loadtxt(input_file)

        if Vbias<Vmin or Vbias>Vmax: continue

        print( f"###  Vbias: {Vbias} file {input_file}" )
        params['VBias'] = Vbias*1000.0
        
        if nThreads == 1:
            positions_, currents = calculate_current(params, input_data)
        else:
            positions_, currents = calculate_current_parallel(params, input_data, nThreads=nThreads)
        if positions is None: positions = positions_

        # Store results
        eps_max = np.maximum.reduce([input_data[:, 3], input_data[:, 4], input_data[:, 5]]) * 1000.0
        eps_max_grid.append(eps_max)
        current_grid.append(currents)

        bias_voltages.append(Vbias)
        
    # Convert lists to numpy arrays
    bias_voltages = np.array(bias_voltages)
    eps_max_grid  = np.array(eps_max_grid)
    current_grid  = np.array(current_grid)

    return bias_voltages, positions, eps_max_grid, current_grid

def eval_dir_of_lines_cpp(input_files, params, Vmin=0.0, Vmax=10.0):
    """Run C++ Pauli simulation for multiple bias voltages"""
    # Initialize arrays
    bias_voltages = []
    positions = None
    eps_max_grid = []
    current_grid = []
    
    for input_file in input_files:
        # Extract bias voltage from filename
        filename = input_file.split('/')[-1]
        if not '_line_scan.dat' in filename: continue
        Vbias = float(filename.split('_line_scan.dat')[0])
        input_data = np.loadtxt(input_file)

        if Vbias < Vmin or Vbias > Vmax: continue
        
        print(f"###  Vbias: {Vbias} file {input_file}")
        params['VBias'] = Vbias*1000.0
        
        # Run C++ simulation
        currents = run_cpp_scan(params, input_data)
        #if positions is None:   positions = input_data[:,:3]
        positions = input_data[:, 0]
        
        # Store results
        eps_max = np.maximum.reduce([input_data[:,3], input_data[:,4], input_data[:,5]]) * 1000.0
        eps_max_grid.append(eps_max)
        current_grid.append(currents)
        bias_voltages.append(Vbias)
    
    # Convert lists to numpy arrays
    bias_voltages = np.array(bias_voltages)
    eps_max_grid = np.array(eps_max_grid)
    current_grid = np.array(current_grid)
    
    return bias_voltages, positions, eps_max_grid, current_grid

def run_cpp_scan(params, input_data):
    """Run C++ Pauli simulation for current calculation"""
    NSingle = int(params['NSingle'])
    NLeads = 2
    
    # Get parameters
    W = params['W']
    VBias = params['VBias']
    Temp = params['Temp']
    VS = np.sqrt(params['GammaS']/np.pi)
    VT = np.sqrt(params['GammaT']/np.pi)
    
    # Initialize solver
    pauli = PauliSolver(NSingle, NLeads)
    
    # Set up leads
    pauli.set_lead(0, 0.0, Temp)  # Substrate lead (mu=0)
    pauli.set_lead(1, VBias, Temp)  # Tip lead (mu=VBias)
    
    # Set up tunneling
    TLeads = np.array([
        [VS, VS, VS],  # Substrate coupling
        [VT, VT, VT]   # Tip coupling
    ])
    pauli.set_tunneling(TLeads)
    
    # Prepare single-particle Hamiltonians
    eps = np.column_stack((input_data[:,3], input_data[:,4], input_data[:,5])) * 1000.0
    npoints = len(eps)
    hsingles = np.zeros((npoints, 3, 3))
    hsingles[:, np.arange(3), np.arange(3)] = eps
    
    # Set up other parameters
    Ws = np.full(npoints, W)
    VGates = np.zeros((npoints, NLeads))
    state_order = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int32)
    
    # Run scan
    currents = pauli.scan_current(
        hsingles=hsingles,
        Ws=Ws,
        VGates=VGates,
        state_order=state_order
    )
    
    return currents

positions = None

# try:
#     # load from .npz file
#     data = np.load('results.npz')
#     bias_voltages = data['bias_voltages']
#     positions     = data['positions']
#     eps_max_grid  = data['eps_max_grid']
#     current_grid  = data['current_grid']
# except FileNotFoundError:
#     # if file not found, calculate results
#     pass
    
if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Run and plot charge transport simulations')
    parser.add_argument('--solver', choices=['qmeq', 'cpp', 'both'], default='qmeq',  help='Which solver to use (qmeq or cpp)')
    parser.add_argument('--Vmin', type=float, default=0.0, help='Minimum bias voltage')
    parser.add_argument('--Vmax', type=float, default=5.5, help='Maximum bias voltage')
    args = parser.parse_args()

    #args.solver = 'C++'
    #args.solver = 'qmeq'

    args.Vmin = 0.0
    args.Vmax = 0.2

    args.Vmin = 0.08
    args.Vmax = 0.12


    input_files = [ 'input/0.10_line_scan.dat']
    Is2=None
    if positions is None:
        if args.solver == 'qmeq' or args.solver == 'both':
            bias_voltages, positions, eps_max_grid, current_grid = eval_dir_of_lines(  input_files, params, Vmin=args.Vmin, Vmax=args.Vmax )
        elif args.solver == 'cpp':
            # Use the new C++ evaluation function
            bias_voltages, positions, eps_max_grid, current_grid = eval_dir_of_lines_cpp( input_files, params, Vmin=args.Vmin, Vmax=args.Vmax)
        if args.solver == 'both':
            _, _, _, Is2 = eval_dir_of_lines_cpp( input_files, params, Vmin=args.Vmin, Vmax=args.Vmax)

    np.savez('results.npz', bias_voltages=bias_voltages, positions=positions, eps_max_grid=eps_max_grid, current_grid=current_grid, params=params)
    
    nV = len(bias_voltages)
    if nV>1:
        didv_grid = calculate_didv(positions, bias_voltages, current_grid)
        plot_results(positions, bias_voltages, eps_max_grid, current_grid, didv_grid )
    elif nV==1:
        didv_grid = None
        print( "bias_voltages.shape: ", bias_voltages.shape )
        print( "current_grid.shape: ", current_grid.shape )
        print( "eps_max_grid.shape: ", eps_max_grid.shape )
        print( "positions.shape: ", positions.shape )

        plot_results_1d(positions, eps_max_grid[0], current_grid[0], Is2=Is2[0], labels=['QmeQ Pauli', 'C++ Pauli'])        
    else:
        print( "Not enough bias voltages to calculate didv" )
        
    plt.savefig( 'run_and_plot.png' )
    plt.show()
    plt.close()