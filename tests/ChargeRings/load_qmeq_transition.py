import numpy as np
import sys
import os

def get_charge(label):
    """
    Calculates the charge of a state from its label string (e.g., '|110>').
    """
    try:
        # Extracts digits from inside the |...> and sums them
        return sum(int(c) for c in label[1:-1])
    except (ValueError, TypeError):
        return 0

def parse_tba_file(filename):
    """
    Parses a Tba.dat file to find non-zero transitions from lower to higher charge
    states, printing the transition value and energy difference. The output is
    sorted alphabetically by the lower-charge state.
    """
    print(f"Parsing file: {filename}")
    with open(filename, 'r') as f:
        header_line = f.readline().strip()
        labels = header_line.split()
        energies_line = f.readline().strip()
        energies_vals = np.array([float(x) for x in energies_line.split()])
        f.readline()
        matrix_data = []
        for line in f:
            numeric_part = line.split('|')[0].strip()
            if numeric_part:
                row = np.array([float(x) for x in numeric_part.split()])
                matrix_data.append(row)
    matrix = np.array(matrix_data)
    n = len(labels)
    if matrix.shape != (n, n):
        print(f"Error: Matrix shape {matrix.shape} does not match the number of labels ({n}).")
        return

    # Create a dictionary to map labels to their energies for easy lookup
    energy_map = {label: energy for label, energy in zip(labels, energies_vals)}

    transitions = []
    for i in range(n):
        for j in range(i):
            value = matrix[i, j]
            if not np.isclose(value, 0.0):
                label_i = labels[i]
                label_j = labels[j]

                charge_i = get_charge(label_i)
                charge_j = get_charge(label_j)

                # Only consider transitions between states of different charge
                if charge_i != charge_j:
                    # Determine which state has lower/higher charge
                    if charge_i > charge_j:
                        low_charge_label, high_charge_label = label_j, label_i                        
                        energy_low, energy_high = energy_map[label_j], energy_map[label_i]
                    else:
                        low_charge_label, high_charge_label = label_i, label_j                        
                        energy_low, energy_high = energy_map[label_i], energy_map[label_j]
                    
                    energy_diff = energy_high - energy_low
                    transitions.append((low_charge_label, high_charge_label, value, energy_diff))

    # Sort transitions alphabetically by the lower-charge state, then the higher-charge state
    sorted_transitions = sorted(transitions, key=lambda x: (x[0], x[1]))

    # Print header for the new output format
    print(f"{'Value':>16s} {'dE [meV]':>16s} \t{'From State':<15} {'To State':<15}")
    print("-" * 70)
    for low_label, high_label, value, dE in sorted_transitions:
        print(f"{value: 16.8f} {dE: 16.4f} \t{low_label:<15} {high_label:<15}")

if __name__ == "__main__":
    files=[
        "/home/prokop/git/ppafm/tests/ChargeRings/790mV/x0.0/790.909dIdV_line_Tba.dat",
        "/home/prokop/git/ppafm/tests/ChargeRings/790mV/x=2.0/790.909dIdV_line_Tba.dat",
        "/home/prokop/git/ppafm/tests/ChargeRings/790mV/x=7.07/790.909dIdV_line_Tba.dat",
        "/home/prokop/git/ppafm/tests/ChargeRings/790mV/x=10.0/790.909dIdV_line_Tba.dat"
    ]
    for filename in files:
        parse_tba_file(filename)
        print("\n") # Add a newline for better separation between file outputs

    # if len(sys.argv) > 1: filename = sys.argv[1]
    # if not os.path.exists(filename): print(f"Error: File not found at {filename}")
    # parse_tba_file(filename)
