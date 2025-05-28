import pauli_scan as ps

def example_xV_scan_with_exp_ref():
    """
    Example showing how to use sweep_scan_param_pauli_xV_orb with experimental reference data.
    """
    from exp_utils import load_and_extract_experimental_data
    
    # Load experimental data
    exp_p0=(9.72, -6.96)
    exp_p1=(-11.0, 15.0)
    exp_STM, exp_dIdV, exp_dist, exp_biases = load_and_extract_experimental_data( filename='exp_rings_data.npz',  start_point=exp_p0, end_point=exp_p1, pointPerAngstrom=5, verbosity=1 )
    ExpRef = { 'STM': exp_STM, 'dIdV': exp_dIdV, 'x': exp_dist, 'voltages': exp_biases}
    
    # Set up base parameters from param_specs
    params = {
        'nsite':   3,
        'radius':  6.2,
        'phiRot':  1.3,
        'phi0_ax': 0.2,
        'VBias':   0.70,
        'Rtip':    2.0,
        'z_tip':   3.0,
        'zV0':    -0.75,
        'zVd':     8.0,
        'zQd':     0.0,
        'Q0':      1.0,
        'Qzz':     0.0,
        'Esite':   -0.080,
        'W':       0.02,
        'decay':   0.3,
        'GammaS':  0.01,
        'GammaT':  0.01,
        'Temp':    0.224,
        'p1_x':    9.72,
        'p1_y':   -9.96,
        'p2_x':   -11.0,
        'p2_y':    12.0
    }

    view_params=['Rtip','z_tip','Esite','zV0','zVd','decay','W','Qzz']
    
    # Define parameter sweep - scanning Rtip and z_tip
    scan_params = [
        #('Rtip',  [2.0,2.5, 3.0, 3.5,4.0]),
        #('p1_y',  [-11.0,-10.5,-10.0,-9.5, -9.0]),

        #('Qzz',  [-20.,-15., -10.0, -5.0, 0.0, 5.0, 10.0, 15.,20.]),
        #('z_tip', [4.0, 4.5, 5.0, 5.5, 6.0])
        #('Esite', [ -0.080,-0.100, -0.120, -0.140 ]),
        #('zVd',  [ 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0 ]),
        #('zV0', [ -0.5, -0.75, -1.0, -1.5, -2.0 ]),
        #('zV0', [ -0.5, -0.5, -0.5, -0.5, -0.5 ]),

        ('zV0',   [ -0.4, -0.5, -0.6,  -0.75,  -1.0  ]),
        #('Esite', [ -0.050, -0.060, -0.070, -0.080, -0.090, -0.100 ]),
        #('decay', [0.1, 0.2, 0.3, 0.4])
    ]
    
    # Run the xV scan with orbital data and experimental reference
    fig, results = ps.sweep_scan_param_pauli_xV_orb( 
        params=params, scan_params=scan_params, view_params=view_params,
        ExpRef=ExpRef, # pointPerAngstrom=5,
        #verbosity=1,  
        nx=100, nV=100, Vmin=0.0, Vmax=1.0,
        result_dir='pauli_scan_results'
    )
    
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    example_xV_scan_with_exp_ref()
