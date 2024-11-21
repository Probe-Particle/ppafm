import numpy as np
from LandauerQD_py import LandauerQDs as LandauerQDs_py
import sys
sys.path.append("../../")
sys.path.append("/home/prokop/git/ppafm")
from pyProbeParticle import LandauerQD as lqd

np.set_printoptions(linewidth=200)

# ===== Test Parameters =====
TOLERANCE = 1e-8

# System parameters (single point test)
nsite = 3
R = 5.0
Q_tip = 0.6
z_tip = 6.0
K = 0.01    # Coulomb interaction between QDs
tS = 0.1    # QD-substrate coupling
tA = 0.1    # Tip coupling strength
decay = 0.7
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening
E_sub = 0.0
E_tip = 0.0
eta = 0.01

# Energy of states on the sites
E0QDs = np.array([-1.0, -1.0, -1.0])

# Single test point for tip position
tip_pos = np.array([0.0, 0.0, z_tip])

def print_comparison(name, py_val, cpp_val, tol=1e-8, ignore_imag=False, relative=False):
    """Print comparison between Python and C++ values."""
    print(f"\n=== {name} ===")
    print("Python implementation:")
    print(py_val)
    print("\nC++ implementation:")
    print(cpp_val)
    
    # Calculate differences
    diff = np.abs(py_val - cpp_val)
    real_diff = np.abs(np.real(py_val) - np.real(cpp_val))
    imag_diff = np.abs(np.imag(py_val) - np.imag(cpp_val))
    
    max_diff = np.max(diff)
    max_real_diff = np.max(real_diff)
    max_imag_diff = np.max(imag_diff)
    
    if relative:
        # Use relative difference for transmission values
        max_diff = max_diff / (np.abs(py_val).mean() + 1e-10)
        max_real_diff = max_real_diff / (np.abs(np.real(py_val)).mean() + 1e-10)
        if not ignore_imag:
            max_imag_diff = max_imag_diff / (np.abs(np.imag(py_val)).mean() + 1e-10)
    
    print(f"\nMax difference: {max_diff}")
    print(f"Max real difference: {max_real_diff}")
    if not ignore_imag:
        print(f"Max imag difference: {max_imag_diff}")
    
    # For full Hamiltonian test, ignore imaginary part differences
    if ignore_imag:
        return max_real_diff < tol
    else:
        return max_diff < tol

def compare_matrix_files(py_file, cpp_file, tol=1e-8):
    """Compare matrices saved in two files."""
    def read_matrix(filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                # Skip title and dimension lines
                data_lines = [line.strip() for line in lines[3:] if line.strip()]
                matrix = []
                for line in data_lines:
                    row = []
                    elements = line.split()
                    for elem in elements:
                        try:
                            # Remove parentheses and split real/imag parts
                            elem = elem.strip('()')
                            if not elem:  # Skip empty elements
                                continue
                            real, imag = map(float, elem.split(','))
                            row.append(complex(real, imag))
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse element '{elem}' in {filename}")
                            continue
                    if row:  # Only add non-empty rows
                        matrix.append(row)
                return np.array(matrix) if matrix else None
        except FileNotFoundError:
            print(f"Warning: File {filename} not found")
            return None
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            return None
    
    py_matrix = read_matrix(py_file)
    cpp_matrix = read_matrix(cpp_file)
    
    if py_matrix is None or cpp_matrix is None:
        print(f"Could not compare matrices - one or both files invalid")
        return False
    
    if py_matrix.shape != cpp_matrix.shape:
        print(f"Matrix shapes differ: {py_matrix.shape} vs {cpp_matrix.shape}")
        return False
    
    diff = np.abs(py_matrix - cpp_matrix)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    print(f"\nComparing {py_file} vs {cpp_file}:")
    print(f"Maximum difference: {max_diff:.2e}")
    print(f"Average difference: {avg_diff:.2e}")
    print(f"Matrices {'match' if max_diff < tol else 'differ significantly'}")
    
    return max_diff < tol

def save_matrix(matrix, filename, title="Matrix"):
    """Save a complex matrix to a file with proper formatting."""
    with open(filename, 'w') as f:
        f.write(f"{title}\n")
        f.write(f"Dimensions: {matrix.shape}\n")
        f.write("Format: (real,imag)\n")
        if len(matrix.shape) == 1:
            # Handle 1D array (vector)
            for elem in matrix:
                f.write(f"({elem.real:.6e},{elem.imag:.6e}) ")
            f.write("\n")
        else:
            # Handle 2D array (matrix)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    elem = matrix[i,j]
                    f.write(f"({elem.real:.6e},{elem.imag:.6e}) ")
                f.write("\n")

def matrices_match(py_matrix, cpp_matrix, tol=1e-6, verbose=False):
    """Check if two matrices match within a tolerance."""
    if py_matrix.shape != cpp_matrix.shape:
        if verbose:
            print("Matrix shapes don't match:", py_matrix.shape, "vs", cpp_matrix.shape)
        return False
    
    diff = np.abs(py_matrix - cpp_matrix)
    max_diff = np.max(diff)
    
    if verbose and max_diff > tol:
        print("Matrices differ by more than {}: max difference = {}".format(tol, max_diff))
        print("\nPython matrix:")
        print(py_matrix)
        print("\nC++ matrix:")
        print(cpp_matrix)
        print("\nDifference matrix:")
        print(diff)
        
        # Find position of maximum difference
        max_pos = np.unravel_index(np.argmax(diff), diff.shape)
        print("\nMaximum difference at position {}:".format(max_pos))
        print("Python value:", py_matrix[max_pos])
        print("C++ value:", cpp_matrix[max_pos])
    
    return max_diff <= tol

def read_matrix_from_file(filename):
    """Read a complex matrix from a file."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Skip title and dimension lines
            data_lines = [line.strip() for line in lines[3:] if line.strip()]
            matrix = []
            for line in data_lines:
                row = []
                elements = line.split()
                for elem in elements:
                    try:
                        # Remove parentheses and split real/imag parts
                        elem = elem.strip('()')
                        if not elem:  # Skip empty elements
                            continue
                        real, imag = map(float, elem.split(','))
                        row.append(complex(real, imag))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse element '{elem}' in {filename}")
                        continue
                if row:  # Only add non-empty rows
                    matrix.append(row)
            return np.array(matrix) if matrix else None
    except FileNotFoundError:
        print(f"Warning: File {filename} not found")
        return None
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return None

def compare_intermediate_matrices():
    """Compare intermediate matrices from Python and C++ calculations."""
    print("\n=== Comparing Intermediate Matrices ===")
    
    # List of matrix pairs to compare (Python file, C++ file)
    matrix_pairs = [
        ("py_G_dag.txt", "cpp_G_dag.txt"),
        ("py_Gamma_t_G_dag.txt", "cpp_Gamma_t_G_dag.txt"),
        ("py_G_Gamma_t_G_dag.txt", "cpp_G_Gamma_t_G_dag.txt"),
        ("py_final.txt", "cpp_final.txt")
    ]
    
    for py_file, cpp_file in matrix_pairs:
        print(f"\nComparing {py_file} vs {cpp_file}:")
        try:
            py_matrix = read_matrix_from_file(py_file)
            cpp_matrix = read_matrix_from_file(cpp_file)
            
            # Print both matrices for visual inspection
            print("\nPython matrix:")
            print(py_matrix)
            print("\nC++ matrix:")
            print(cpp_matrix)
            
            # Calculate and print differences
            diff = np.abs(py_matrix - cpp_matrix)
            max_diff = np.max(diff)
            avg_diff = np.mean(diff)
            print(f"\nMaximum absolute difference: {max_diff}")
            print(f"Average absolute difference: {avg_diff}")
            
            # Print locations of largest differences
            if max_diff > 1e-10:
                large_diff_mask = diff > max_diff * 0.1
                print("\nLocations of large differences:")
                for i, j in zip(*np.where(large_diff_mask)):
                    print(f"Position ({i},{j}): Python={py_matrix[i,j]}, C++={cpp_matrix[i,j]}, Diff={diff[i,j]}")
        
        except FileNotFoundError as e:
            print(f"Could not find file: {e.filename}")
        except Exception as e:
            print(f"Error comparing matrices: {e}")

def run_tests():
    """Test Green's function calculation step by step"""
    print("\n=== Test 4: Debugging Green's function and transmission calculation ===")
    
    # Initialize QD system
    n_qds = 3
    qd_pos = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ])
    e_sites = np.array([1.0, 1.0, 1.0])
    
    # Test parameters
    tip_pos = np.array([1.0, 0.0, 1.0])
    energy = 1.0
    eta = 1e-6  # Small imaginary part for Green's function
    
    # Initialize both implementations
    py_system = LandauerQDs_py(qd_pos, e_sites, K=K, decay=decay, tS=tS, E_sub=E_sub, E_tip=E_tip, tA=tA, eta=eta, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    lqd.init(qd_pos, e_sites, K=K, decay=decay, tS=tS, E_sub=E_sub, E_tip=E_tip, tA=tA, eta=eta, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    print("\nStep 1: Get Hamiltonian without tip")
    Hqd_py  = py_system.H_QD_no_tip;  print("Hqd_py.shape:",  Hqd_py.shape)
    Hqd_cpp = lqd.get_H_QD_no_tip();  print("Hqd_cpp.shape:", Hqd_cpp.shape)
    print("Max difference in H_QD:", np.max(np.abs(Hqd_py - Hqd_cpp)))
    #save_matrix(Hqd_py, "py_H_QD_no_tip.txt", "H_QD_no_tip (Python)")
    #save_matrix(Hqd_cpp, "cpp_H_QD_no_tip.txt", "H_QD_no_tip (C++)")
    if not matrices_match(Hqd_py, Hqd_cpp, verbose=True):
        print("ERROR: H_QD matrices don't match!")
        return False
    
    print("\nStep 2: Get full Hamiltonian with tip")
    H_py  = py_system.make_full_hamiltonian(tip_pos, Q_tip=Q_tip) 
    H_cpp = lqd.get_full_H(tip_pos)
    print("Max difference in full H:", np.max(np.abs(H_py - H_cpp)))
    #save_matrix(H_py,  "py_H.txt",  "H_full (Python)")
    #save_matrix(H_cpp, "cpp_H.txt", "H_full (C++)")
    if not matrices_match(H_py, H_cpp, verbose=True):
        print("ERROR: Full H matrices don't match!")
        return False

    N = n_qds+2

    print("\nStep 3: Construct (EI - H) matrix")
    # Python calculation
    identity = np.eye(N, dtype=np.complex128)
    A_py  = (energy + 1j*eta)*identity - H_py
    A_cpp = (energy + 1j*eta)*identity - H_cpp
    
    # C++ calculation - we'll save the pre-inversion matrix
    G_cpp = np.zeros((N,N), dtype=np.complex128)
    lqd.calculate_greens_function(energy, H_cpp, G_cpp)  # This saves pre-inversion matrix to cpp_A.txt and cpp_G.txt
    
    # Load the C++ pre-inversion matrix from file
    A_cpp_ = read_matrix_from_file("cpp_pre_inversion.txt")
    
    print("\nPython A = (EI - H) :")
    print(A_py)
    print("\nC++    A = (EI - H) :")
    print(A_cpp)
    print("\nMax difference in (EI - H):", np.max(np.abs( A_py - A_cpp)))
    
    if not matrices_match(A_py, A_cpp, verbose=True):
        print("ERROR: (EI - H) matrices don't match!")
        return False
    
    print("\nStep 4: Calculate Green's function G = (EI - H)^(-1)")
    # Python calculation
    G_py = np.linalg.inv(A_py)
    
    # C++ calculation
    G_cpp = np.zeros((N, N), dtype=np.complex128)
    lqd.calculate_greens_function(energy, H_cpp, G_cpp)  # This performs the full calculation including inversion
    
    print("\nPython Green's function G:")
    print(G_py)
    print("\nC++ Green's function G:")
    print(G_cpp)
    print("\nMax difference in G:", np.max(np.abs(G_py - G_cpp)))
    
    save_matrix(G_py, "py_G.txt", "G (Python)")
    save_matrix(G_cpp, "cpp_G.txt", "G (C++)")
    
    if not matrices_match(G_py, G_cpp, verbose=True):
        print("ERROR: Green's functions don't match!")
        return False
    
    # Verify that G is actually the inverse
    print("\nVerification that G is the inverse:")
    verify_py = np.matmul(A_py, A_py)
    print("Python G verification (should be identity):")
    print(verify_py)
    verify_cpp = np.matmul(A_cpp, A_cpp)
    print("C++ G verification (should be identity):")
    print(verify_cpp)
    
    if not matrices_match(verify_py, verify_cpp, tol=1e-6, verbose=True):  # Use higher tolerance for verification
        print("ERROR: G verification matrices don't match!")
        return False

    print("\nStep 5: Calculate transmission")
    # Run transmission calculation
    T_py = py_system._calculate_transmission_from_H(H_py, energy)
    
    # Prepare arrays for C++ call
    tip_pos_arr  = np.array([tip_pos], dtype=np.float64)  # Shape: (1, 3)
    energies_arr = np.array([energy], dtype=np.float64)  # Shape: (1,)
    Hqds_cpp     = np.ascontiguousarray( Hqd_cpp.reshape(-1), dtype=np.complex128)  # Flatten for C++
    
    T_cpp = lqd.calculate_transmissions(tip_pos_arr, energies_arr, Hqds_cpp)[0,0]
    
    print("\nPython transmission:", T_py)
    print("C++ transmission:",      T_cpp)
    print("Relative difference:", abs(T_py - T_cpp) / (abs(T_py) + 1e-10))
    
    # Compare intermediate matrices
    compare_intermediate_matrices()
    
    # Compare gamma matrices
    print("\n=== Comparing Gamma Matrices ===")
    Gamma_s_py, Gamma_t_py = py_system._calculate_coupling_matrices()
    
    print("\nPython Gamma_s:")
    print(Gamma_s_py)
    print("\nPython Gamma_t:")
    print(Gamma_t_py)
    
    # Load C++ gamma matrices from files
    try:
        Gamma_s_cpp = read_matrix_from_file("cpp_Gamma_s.txt")
        Gamma_t_cpp = read_matrix_from_file("cpp_Gamma_t.txt")
        
        print("\nC++ Gamma_s:")
        print(Gamma_s_cpp)
        print("\nC++ Gamma_t:")
        print(Gamma_t_cpp)
        
        # Calculate differences
        diff_s = np.abs(Gamma_s_py - Gamma_s_cpp)
        diff_t = np.abs(Gamma_t_py - Gamma_t_cpp)
        
        print("\nMaximum difference in Gamma_s:", np.max(diff_s))
        print("Maximum difference in Gamma_t:", np.max(diff_t))
    except FileNotFoundError as e:
        print(f"Could not find gamma matrix file: {e.filename}")
    except Exception as e:
        print(f"Error comparing gamma matrices: {e}")
    
    if abs(trans_py - trans_cpp) / (abs(trans_py) + 1e-10) > 1e-6:
        print("ERROR: Transmissions don't match!")
        return False
        
    print("\nAll tests passed successfully!")
    
    sys.exit(0)  # Exit after completing the tests
    return True

if __name__ == "__main__":
    run_tests()  # Only run this test
