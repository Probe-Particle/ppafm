import numpy as np
import sys

def print_comparison_complex(name, A, B, tol=1e-8, ignore_imag=False, relative=False):
    """Print comparison between Python and C++ values."""
    print(f"\n=== {name} ===")
    print("Python implementation:")
    print(A)
    print("\nC++ implementation:")
    print(B)
    
    # Calculate differences
    diff = np.abs(A - B)
    real_diff = np.abs(np.real(A) - np.real(B))
    imag_diff = np.abs(np.imag(A) - np.imag(B))
    
    max_diff = np.max(diff)
    max_real_diff = np.max(real_diff)
    max_imag_diff = np.max(imag_diff)
    
    if relative:
        # Use relative difference for transmission values
        max_diff = max_diff / (np.abs(A).mean() + 1e-10)
        max_real_diff = max_real_diff / (np.abs(np.real(A)).mean() + 1e-10)
        if not ignore_imag:
            max_imag_diff = max_imag_diff / (np.abs(np.imag(A)).mean() + 1e-10)
    
    print(f"\nMax difference: {max_diff}")
    print(f"Max real difference: {max_real_diff}")
    if not ignore_imag:
        print(f"Max imag difference: {max_imag_diff}")
    
    # For full Hamiltonian test, ignore imaginary part differences
    if ignore_imag:
        return max_real_diff < tol
    else:
        return max_diff < tol

def compare_matrix_files(A_file, B_file, tol=1e-8):
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
    
    A  = read_matrix(A_file)
    B = read_matrix(B_file)
    
    if A is None or B is None:
        print(f"Could not compare matrices - one or both files invalid")
        return False
    
    if A.shape != B.shape:
        print(f"Matrix shapes differ: {A.shape} vs {B.shape}")
        return False
    
    diff = np.abs(A - B)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    print(f"\nComparing {A_file} vs {B_file}:")
    print(f"Maximum difference: {max_diff:.2e}")
    print(f"Average difference: {avg_diff:.2e}")
    print(f"Matrices {'match' if max_diff < tol else 'differ significantly'}")
    
    return max_diff < tol

# def save_matrix(matrix, filename, title="Matrix"):
#     """Save a complex matrix to a file with proper formatting."""
#     with open(filename, 'w') as f:
#         f.write(f"{title}\n")
#         f.write(f"Dimensions: {matrix.shape}\n")
#         f.write("Format: (real,imag)\n")
#         if len(matrix.shape) == 1:
#             # Handle 1D array (vector)
#             for elem in matrix:
#                 f.write(f"({elem.real:.6e},{elem.imag:.6e}) ")
#             f.write("\n")
#         else:
#             # Handle 2D array (matrix)
#             for i in range(matrix.shape[0]):
#                 for j in range(matrix.shape[1]):
#                     elem = matrix[i,j]
#                     f.write(f"({elem.real:.6e},{elem.imag:.6e}) ")
#                 f.write("\n")

def save_matrix(matrix, filename=None, title="Matrix"):
    """Save a complex matrix to a file or standard output with proper formatting."""
    f = open(filename, 'w') if filename else sys.stdout  # Use stdout if filename is None
    # Write the content
    f.write(f"{title} Dimensions: {matrix.shape} Format: (real,imag) \n")
    if len(matrix.shape) == 1:  # Handle 1D array (vector)
        f.write(" ".join( f"({elem.real:.6e},{elem.imag:.6e})" for elem in matrix) + "\n")
    else:  # Handle 2D array (matrix)
        for i in range(matrix.shape[0]):
            f.write(" ".join( f"({matrix[i, j].real:.6e},{matrix[i, j].imag:.6e})" for j in range(matrix.shape[1]) ) + "\n")
    if f is not sys.stdout:  # Close the file only if it's not stdout
        f.close()

def matrices_match(A, B, tol=1e-6, verbose=False):
    """Check if two matrices match within a tolerance."""
    if A.shape != B.shape:
        if verbose:
            print("Matrix shapes don't match:", A.shape, "vs", B.shape)
        return False
    
    diff = np.abs(A - B)
    max_diff = np.max(diff)
    
    if verbose and max_diff > tol:
        print("Matrices differ by more than {}: max difference = {}".format(tol, max_diff))
        print("\nPython matrix:")
        print(A)
        print("\nC++ matrix:")
        print(B)
        print("\nDifference matrix:")
        print(diff)
        
        # Find position of maximum difference
        max_pos = np.unravel_index(np.argmax(diff), diff.shape)
        print("\nMaximum difference at position {}:".format(max_pos))
        print("Python value:", A[max_pos])
        print("C++ value:", B[max_pos])
    
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
