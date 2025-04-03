import numpy as np
import matplotlib.pyplot as plt

import glob
from current import calculate_current, calculate_didv, load_parameters
from plot_utils import plot_results

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

def eval_dir_of_lines( input_files, params, Vmin = 0.0, Vmax = 10.0  ):
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
        
        positions_, currents = calculate_current(params, input_data)
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

positions = None
try:
    # load from .npz file
    data = np.load('results.npz')
    bias_voltages = data['bias_voltages']
    positions     = data['positions']
    eps_max_grid  = data['eps_max_grid']
    current_grid  = data['current_grid']
except FileNotFoundError:
    # if file not found, calculate results
    pass
    
if positions is None:
    bias_voltages, positions, eps_max_grid, current_grid = eval_dir_of_lines( input_files, params, Vmin = 0.00, Vmax = 5.5  )
    np.savez('results.npz', bias_voltages=bias_voltages, positions=positions, eps_max_grid=eps_max_grid, current_grid=current_grid, params=params )

didv_grid = calculate_didv(positions, bias_voltages, current_grid)

plot_results(positions, bias_voltages, eps_max_grid, current_grid, didv_grid )
plt.show()
plt.savefig( 'qmeq_results_summary.png' )
plt.close()