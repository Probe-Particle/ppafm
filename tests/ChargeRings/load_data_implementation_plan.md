# Implementation Plan for Loading Data Back to GUI

## Current State Analysis

Currently, the `save_everything` function in `CombinedChargeRingsGUI_v5.py` saves:

1. **Parameters (JSON)**: All GUI parameters via `get_param_values()` 
2. **Visualization (PNG)**: Screenshot of the current plots
3. **Simulation Data (NPZ)**: Data structures including:
   - `spos`: Site positions
   - `STM`: STM current data
   - `Es`: Site energies 
   - `Ts`: Tunneling rates
   - `probs`: Probabilities of different states
   - `params_json`: Same parameters as in the JSON file

## Decoupling Strategy

To implement loading functionality, we should reorganize the code to separate:

1. **Data Generation**: Calculation of simulation results
2. **Visualization**: Plotting the results
3. **State Management**: Handling GUI parameter state

## Implementation Plan

### 1. Code Refactoring

#### a. Create a Simulation State Class

```python
class SimulationState:
    def __init__(self):
        self.params = {}
        self.spos = None
        self.STM = None
        self.dIdV = None
        self.Es = None
        self.Ts = None
        self.probs = None
        self.rots = None
        
    def from_npz(self, npz_file):
        # Load from .npz file
        data = np.load(npz_file)
        self.STM = data['STM']
        self.Es = data['Es']
        self.Ts = data['Ts']
        self.probs = data['probs']
        self.spos = data['spos']
        # Load params from params_json
        if 'params_json' in data:
            self.params = json.loads(data['params_json'])
        return self
        
    def to_npz(self, npz_file):
        # Save to .npz file (same as current save_everything)
        data = {
            'spos': self.spos,
            'STM': self.STM,
            'Es': self.Es,
            'Ts': self.Ts,
            'probs': self.probs,
            'params_json': json.dumps(self.params)
        }
        np.savez(npz_file, **data)
        # Also save JSON separately
        json_file = os.path.splitext(npz_file)[0] + '.json'
        with open(json_file, 'w') as f:
            json.dump(self.params, f, indent=4)
```

#### b. Extract Plotting Functions from the Main GUI Class

Currently, plotting is mixed with calculation in methods like `run()`. We should separate these:

```python
class PlottingManager:
    def __init__(self, axes):
        self.axes = axes  # Dictionary of matplotlib axes
        
    def plot_simulation_results(self, sim_state):
        # Plot site energies
        self.plot_site_energies(sim_state.Es, sim_state.spos, ax=self.axes['ax1'])
        # Plot STM current
        self.plot_stm_current(sim_state.STM, ax=self.axes['ax5'])
        # Plot dI/dV
        self.plot_didv(sim_state.dIdV, ax=self.axes['ax6'])
        # etc...
        
    def plot_site_energies(self, Es, spos, ax):
        # Extracted from current plotting code
        pass
        
    def plot_stm_current(self, STM, ax):
        # Extracted from current plotting code
        pass
```

#### c. Modify the `run()` Method to Use the New Classes

Restructure the `run()` method in `ApplicationWindow` to:
1. Create a SimulationState object
2. Populate it with calculation results
3. Use PlottingManager to visualize

```python
def run(self):
    # Create simulation state
    sim_state = SimulationState()
    sim_state.params = self.get_param_values()
    
    # Clear axes
    self.ax1.cla(); self.ax2.cla(); self.ax3.cla() 
    self.ax4.cla(); self.ax5.cla(); self.ax6.cla()
    
    # Run calculation based on mode
    if self.cbPlotXV.isChecked():
        # xV calculation
        STM, dIdV, Es, Ts, probs, x, Vbiases, spos, rots = self._calculate_xv_data(sim_state.params)
    else:
        # xy calculation
        STM, dIdV, Es, Ts, probs, spos, rots = self._calculate_xy_data(sim_state.params)
    
    # Store results in simulation state
    sim_state.STM = STM
    sim_state.dIdV = dIdV
    sim_state.Es = Es
    sim_state.Ts = Ts
    sim_state.probs = probs
    sim_state.spos = spos
    sim_state.rots = rots
    
    # Use plotting manager to display results
    self.plotting_manager.plot_simulation_results(sim_state)
    
    # Store current simulation state
    self.current_sim_state = sim_state
    
    return STM, dIdV, Es, Ts, probs, spos, rots
```

### 2. Add Load Functionality

#### a. Create a `load_everything()` Method

```python
def load_everything(self):
    """Load NPZ data and parameters"""
    fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Data", "", "NPZ files (*.npz)")
    if not fname:
        return
        
    # Create simulation state from file
    sim_state = SimulationState().from_npz(fname)
    
    # Update GUI parameters
    self.set_param_values(sim_state.params)
    
    # Update plots using the plotting manager
    self.plotting_manager.plot_simulation_results(sim_state)
    
    # Store as current simulation state
    self.current_sim_state = sim_state
```

#### b. Add Load Button to GUI

Add a "Load" button next to the "Save All" button:

```python
btnLoad = QtWidgets.QPushButton("Load")
btnLoad.clicked.connect(self.load_everything)
self.toolBar.addWidget(btnLoad)
```

#### c. Implement `set_param_values()` Method

```python
def set_param_values(self, params):
    """Set GUI parameter values from dictionary"""
    # For each parameter widget, set its value
    for param_name, value in params.items():
        if param_name in self.params:
            widget_info = self.params[param_name]
            widget = widget_info.get('widget_obj')
            if widget is not None:
                # Set value based on widget type
                if isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(bool(value))
                # Add other widget types as needed
```

## Key Variables and Functions

### Important Variables

1. `params`: Dictionary of all simulation parameters
2. `current_sim_state`: Current SimulationState object
3. `STM`, `dIdV`, `Es`, `Ts`, `probs`, `spos`, `rots`: Simulation result arrays

### Important Functions

1. `run()`: Main calculation function
2. `_calculate_xy_data()`: XY plane simulation calculation
3. `_calculate_xv_data()`: XV line simulation calculation
4. `save_everything()`: Save all data
5. `load_everything()`: Load all data
6. `get_param_values()`: Get current parameters
7. `set_param_values()`: Set parameters in GUI

## Testing Plan

1. Test loading parameter values into the GUI
2. Test recreating plots from loaded data
3. Test edge cases: missing data fields, older file formats
4. Test re-running calculation with loaded parameters vs. using cached results
