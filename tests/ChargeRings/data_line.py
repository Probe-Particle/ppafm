def read_calculation_parameters(f, pos=None):
    """
    Read calculation parameters from a file object into a dictionary.
    Parameters are lines starting with '# #' and containing a colon.
    
    Args:
        f: File object (already opened)
        pos: Optional file position to seek to before reading
        
    Returns:
        tuple: (dictionary of parameters, file position after reading)
    """
    if pos is not None:
        f.seek(pos)
        
    params = {}
    
    while True:
        line = f.readline()
        if not line:
            break
            
        line = line.strip()
        if line.startswith('# # Calculation parameters:'):
            # Read parameters until we hit the next section
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    return params, pos
                    
                line = line.strip()
                if line.startswith('# # Data columns:'):
                    return params, pos
                if line.startswith('# #') and ':' in line:
                    # Remove '# #' and split by first colon
                    key_value = line[3:].split(':', 1)
                    key       = key_value[0].strip()
                    value     = key_value[1].strip()
                    
                    # Try to convert to float if possible
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    
                    params[key] = value
    return params, f.tell()

def read_data_columns(f, pos=None):
    """
    Read data column descriptions from a file object.
    
    Args:
        f: File object (already opened)
        pos: Optional file position to seek to before reading
        
    Returns:
        tuple: (list of column descriptions, file position after reading)
    """
    if pos is not None:
        f.seek(pos)
        
    columns = []
    
    while True:
        line = f.readline()
        if not line:
            break
            
        line = line.strip()
        if line.startswith('# # Data columns:'):
            # Read columns until we hit the data section
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    return columns, pos
                    
                line = line.strip()
                if not line.startswith('# #') or ':' not in line:
                    return columns, pos
                
                # Extract column description (everything after colon)
                desc = line.split(':', 1)[1].strip()
                columns.append(desc)
    return columns, f.tell()

def read_data_lines(f, pos=None, nsite=None):
    """
    Read numerical data lines from a file object.
    
    Args:
        f: File object (already opened)
        pos: Optional file position to seek to before reading
        nsite (int, optional): Number of sites from calculation parameters
                             If None, will be determined from file
        
    Returns:
        numpy.ndarray: Array of numerical data
    """
    import numpy as np
    
    if pos is not None:
        f.seek(pos)
    
    # Skip all lines starting with '#'
    data_lines = []
    while True:
        pos = f.tell()
        line = f.readline()
        if not line:
            break
            
        line = line.strip()
        if not line.startswith('#') and line:
            data_lines.append(line)
    
    # Convert to numpy array
    data = np.loadtxt(data_lines)
    return data

def read_dat_file(filepath, bColumDict=True):
    """
    Convenience function to read all sections of a .dat file in one pass.
    
    Args:
        filepath (str): Path to the .dat file
        
    Returns:
        tuple: (params_dict, columns_list, data_array)
    """
    with open(filepath, 'r') as f:
        params, pos  = read_calculation_parameters(f)
        columns, pos = read_data_columns(f, pos)
        data         = read_data_lines(f, pos)
        if bColumDict:
            columns = { columns[i]: i for i in range(len(columns)) }
        return params, columns, data