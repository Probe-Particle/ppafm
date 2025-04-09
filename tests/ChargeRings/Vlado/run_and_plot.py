import numpy as np
import matplotlib.pyplot as plt

import glob
import argparse
from current import calculate_current, calculate_didv, load_parameters, calculate_current_parallel
from plot_utils import plot_results, plot_results_1d
import time

# Add C++ solver imports
import sys
from sys import path
path.insert(0, '../../../pyProbeParticle')
import pauli as pls
#from pauli import PauliSolver

params = load_parameters( config_file='qmeq.in' )

# print( "#### params: " )
# for k, v in params.items(): print(f"{k}: {v}")

params['VS'] = np.sqrt(params['GammaS']/np.pi)
params['VT'] = np.sqrt(params['GammaT']/np.pi)

input_files = glob.glob('input/*.dat')
input_files = sorted(input_files)

print("#### params: ")
for k, v in params.items(): print(f"{k}: {v}")


def limit_input_line( input_data, vmin=1e+300, vmax=-1e+300 ):
    return input_data[(input_data[:,0] >= vmin) & (input_data[:,0] <= vmax)]

def eval_dir_of_lines( input_files, params, Vmin = 0.0, Vmax = 10.0, nThreads=12, line_lims=None  ):
    # Initialize arrays
    bias_voltages = []
    positions     = None
    eps_max_grid  = []
    current_grid  = []

    # Print QmeQ parameters before simulation
    print("\n=== QmeQ Solver Parameters ===")
    print(f"Vmin: {Vmin}, Vmax: {Vmax}")
    print(f"NSingle: {params['NSingle']}, W: {params['W']}")
    print(f"Temp: {params['Temp']}, VBias: {params['VBias']}")
    print(f"GammaS: {params['GammaS']}, GammaT: {params['GammaT']}")
    print(f"VS: {params['VS']}, VT: {params['VT']}")
    VS_qmeq = params['VS']
    VT_qmeq = params['VT']
    # QmeQ-specific parameters from current.py
    print(f"DBand (from params): {params.get('DBand', 'not specified')}")

    for input_file in input_files:
        # Extract bias voltage from filename
        filename = input_file.split('/')[-1]
        if not '_line_scan' in filename: continue
        Vbias = float(filename.split('_')[0])
        input_data = np.loadtxt(input_file)
        
        # Ensure input_data is always 2D
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)

        if line_lims is not None: input_data = limit_input_line( input_data, vmin=line_lims[0], vmax=line_lims[1] )
            

        if Vbias<Vmin or Vbias>Vmax: continue

        print( f"###  Vbias: {Vbias} file {input_file}" )
        params['VBias'] = Vbias*1000.0
        
        # Print a sample of the site energies (first 3 rows)
        if input_data.shape[0] > 3:
            print(f"\nQmeQ Sample input energies (first 3 points):")
            for i in range(3):
                print(f"Point {i}: Esite_1: {input_data[i,3]*1000.0:.3f}, Esite_2: {input_data[i,4]*1000.0:.3f}, Esite_3: {input_data[i,5]*1000.0:.3f} meV")
        elif input_data.shape[0] > 0:
            print(f"\nQmeQ Input energy (single point):")
            print(f"Esite_1: {input_data[0,3]*1000.0:.3f}, Esite_2: {input_data[0,4]*1000.0:.3f}, Esite_3: {input_data[0,5]*1000.0:.3f} meV")
            
        if nThreads == 1:
            positions_, currents = calculate_current(params, input_data, verbosity=verbosity)
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

def eval_dir_of_lines_cpp(input_files, params, Vmin=0.0, Vmax=10.0, line_lims=None):
    """Run C++ Pauli simulation for multiple bias voltages"""
    # Initialize arrays
    bias_voltages = []
    positions = None
    eps_max_grid = []
    current_grid = []
    
    # Print C++ solver parameters before simulation
    print("\n=== C++ Pauli Solver Parameters ===")
    print(f"Vmin: {Vmin}, Vmax: {Vmax}")
    print(f"NSingle: {params['NSingle']}, W: {params['W']}")
    print(f"Temp: {params['Temp']}, VBias: {params['VBias']}")
    print(f"GammaS: {params['GammaS']}, GammaT: {params['GammaT']}")
    VS_cpp = np.sqrt(params['GammaS']/np.pi)
    VT_cpp = np.sqrt(params['GammaT']/np.pi)
    print(f"VS: {VS_cpp}, VT: {VT_cpp} (calculated from GammaS/T)")
    DBand_cpp = params['DBand'] # DBand does not seem to be used in C++ solver ?
    print(f"DBand (hard-coded in qmeq.in): {DBand_cpp}")
    print(f"TunnelMatrix: [[{VS_cpp}, {VS_cpp}, {VS_cpp}], [{VT_cpp}, {VT_cpp}, {VT_cpp}]]")
    
    
    for input_file in input_files:
        # Extract bias voltage from filename
        filename = input_file.split('/')[-1]
        if not '_line_scan' in filename: continue
        Vbias = float(filename.split('_')[0])
        input_data = np.loadtxt(input_file)

        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)

        if line_lims is not None: input_data = limit_input_line( input_data, vmin=line_lims[0], vmax=line_lims[1] )

        if Vbias < Vmin or Vbias > Vmax: continue
        
        #print(f"###  Vbias: {Vbias} file {input_file}")
        params['VBias'] = Vbias * 1000.0

        pls.verbosity = verbosity
        currents = pls.run_cpp_scan(params, input_data[:,3:6]*1000.0, input_data[:,9:12], scaleE=1.0 )

        #if positions is None:   positions = input_data[:,:3]
        positions = input_data[:, 0]
        
        #print( "currents ", currents )

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

positions = None


if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Run and plot charge transport simulations')
    parser.add_argument('--solver', choices=['qmeq', 'cpp', 'both'], default='cpp',  help='Which solver to use (qmeq or cpp)')
    parser.add_argument('--Vmin', type=float, default=0.0, help='Minimum bias voltage')
    parser.add_argument('--Vmax', type=float, default=5.5, help='Maximum bias voltage')
    args = parser.parse_args()

    #args.solver = 'C++'
    #args.solver = 'qmeq'

    args.Vmin = 0.0
    args.Vmax = 0.3

    #args.Vmin = 0.08
    #args.Vmax = 0.12

    #line_lims = [10.0, 30.0]
    line_lims = None    

    #args.solver = 'both'
    verbosity = 0

    # Disable line wrapping entirely
    np.set_printoptions(linewidth=np.inf)

    input_files = [ name for name in input_files if '_line_scan.dat' in name ]
    #input_files = [ 'input/0.10_line_scan.dat']
    #input_files = [ 'input/0.10_line_scan_short.dat']
    #input_files = [ 'input/0.10_line_scan_20.dat']
    Is2=None

    # Nano-time to measure preformance
    T0 = time.perf_counter()
    print( " input_files ",  input_files )
    if positions is None:
        if args.solver == 'qmeq':
            bias_voltages, positions, eps_max_grid, current_grid = eval_dir_of_lines(  input_files, params, Vmin=args.Vmin, Vmax=args.Vmax, line_lims=line_lims, nThreads=12 )
        elif args.solver == 'cpp':
            # Use the new C++ evaluation function
            bias_voltages, positions, eps_max_grid, current_grid = eval_dir_of_lines_cpp( input_files, params, Vmin=args.Vmin, Vmax=args.Vmax, line_lims=line_lims)
        elif args.solver == 'both':
            print( "\n\n####################################################"  )
            print( "#################### QmeQ Pauli #####################"  )
            print( "####################################################\n\n"  )
            #verbosity=0
            bias_voltages, positions, eps_max_grid, current_grid = eval_dir_of_lines(  input_files, params, Vmin=args.Vmin, Vmax=args.Vmax, line_lims=line_lims, nThreads=1 )
            #verbosity=3
            print( "\n\n####################################################"  )
            print( "#################### C++ Pauli #####################"  )
            print( "####################################################\n\n"  )
            _, _, _, Is2 = eval_dir_of_lines_cpp( input_files, params, Vmin=args.Vmin, Vmax=args.Vmax, line_lims=line_lims)
            #bias_voltages, positions, eps_max_grid,Is2 = eval_dir_of_lines_cpp( input_files, params, Vmin=args.Vmin, Vmax=args.Vmax, line_lims=line_lims)
            #current_grid = Is2*0.
            Is2=Is2[0]
    T = time.perf_counter() - T0
    print( "Total time: ", T, "s" )

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
        plot_results_1d(positions, eps_max_grid[0], current_grid[0], Is2=Is2, labels=['QmeQ Pauli', 'C++ Pauli'], ms=5.0 )        
    else:
        print( "Not enough bias voltages to calculate didv" )
        
    plt.savefig( 'run_and_plot.png' )
    plt.show()
    plt.close()