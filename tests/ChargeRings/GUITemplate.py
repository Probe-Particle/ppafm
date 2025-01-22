import sys
import json
from PyQt5 import QtCore, QtWidgets

class GUITemplate(QtWidgets.QMainWindow):
    def __init__(self, title="Application GUI"):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.main_widget = QtWidgets.QWidget(self)
        
        # Initialize parameter specifications
        self.param_specs = {}
        self.param_widgets = {}
        
        # Create GUI
        #self.create_gui()
        
    def create_gui(self):
        """Create the main GUI layout and widgets"""
        # Main layout
        l00 = QtWidgets.QHBoxLayout(self.main_widget)
        
        # Control panel layout
        l0 = QtWidgets.QVBoxLayout()
        l00.addLayout(l0)
        
        # Create widgets for each parameter group
        current_group = None
        current_layout = None
        
        for param_name, spec in self.param_specs.items():
            # Create new group if needed
            if spec['group'] != current_group:
                current_group = spec['group']
                gb = QtWidgets.QGroupBox(current_group)
                l0.addWidget(gb)
                current_layout = QtWidgets.QVBoxLayout(gb)
            
            # Create widget layout
            hb = QtWidgets.QHBoxLayout()
            current_layout.addLayout(hb)
            hb.addWidget(QtWidgets.QLabel(f"{param_name}:"))
            
            # Create appropriate widget type
            if spec['widget'] == 'double':
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(*spec['range'])
                widget.setValue(spec['value'])
                widget.setSingleStep(spec['step'])
                if 'decimals' in spec:
                    widget.setDecimals(spec['decimals'])
            elif spec['widget'] == 'int':
                widget = QtWidgets.QSpinBox()
                widget.setRange(*spec['range'])
                widget.setValue(spec['value'])
                if 'step' in spec:
                    widget.setSingleStep(spec['step'])
            
            widget.valueChanged.connect(self.on_parameter_change)
            hb.addWidget(widget)
            self.param_widgets[param_name] = widget
        
        # Add common controls
        self.add_common_controls(l0)
        
        # Set the central widget
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
    
    def add_common_controls(self, layout):
        """Add common control buttons"""
        # Auto-update checkbox
        hb = QtWidgets.QHBoxLayout()
        layout.addLayout(hb)
        self.cbAutoUpdate = QtWidgets.QCheckBox("Auto-update")
        self.cbAutoUpdate.setChecked(True)
        hb.addWidget(self.cbAutoUpdate)
        
        # Run Button
        btn = QtWidgets.QPushButton("Run")
        btn.clicked.connect(self.run)
        hb.addWidget(btn)
        
        # Save/Load buttons
        hb = QtWidgets.QHBoxLayout()
        layout.addLayout(hb)
        btnSave = QtWidgets.QPushButton("Save Parameters")
        btnSave.clicked.connect(self.save_parameters)
        hb.addWidget(btnSave)
        btnLoad = QtWidgets.QPushButton("Load Parameters")
        btnLoad.clicked.connect(self.load_parameters)
        hb.addWidget(btnLoad)
    
    def get_param_values(self):
        """Get current values of all parameters"""
        return {name: widget.value() for name, widget in self.param_widgets.items()}
    
    def set_param_values(self, values):
        """Set values for all parameters"""
        for name, value in values.items():
            if name in self.param_widgets:
                self.param_widgets[name].setValue(value)
    
    def save_parameters(self):
        """Save parameters to JSON file"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "JSON files (*.json)")
        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            with open(filename, 'w') as f:
                json.dump(self.get_param_values(), f, indent=4)
    
    def load_parameters(self):
        """Load parameters from JSON file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON files (*.json)")
        if filename:
            with open(filename, 'r') as f:
                values = json.load(f)
                self.set_param_values(values)
                self.run()
    
    def on_parameter_change(self):
        """Handle parameter changes"""
        if self.cbAutoUpdate.isChecked():
            self.run()
    
    def run(self):
        """Main execution method to be implemented by child classes"""
        raise NotImplementedError("Child classes must implement run() method")

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = GUITemplate()
    aw.show()
    sys.exit(qApp.exec_())
