# interactive_plotter.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from wasserstein_distance import wasserstein_1d_general, wasserstein_1d_grid # Correct import

class InteractiveWassersteinPlotter:
    def __init__(self,
                 x_coords_domain: np.ndarray, # The common x-coordinates for model evaluation & plotting
                 reference_model_func,       # func(x_coords, params_dict) -> y_values
                 reference_params: dict,
                 interactive_model_func,     # func(x_coords, params_dict) -> y_values
                 interactive_param_configs: dict, # name: (min, max, init, step)
                 use_grid_optimized_wd: bool = True
                ):
        self.x_coords_domain = x_coords_domain # This is the 'xs' for plotting and grid-based W1
        self.reference_model_func = reference_model_func
        self.reference_params = reference_params
        self.interactive_model_func = interactive_model_func
        self.interactive_param_configs = interactive_param_configs
        self.use_grid_optimized_wd = use_grid_optimized_wd

        if use_grid_optimized_wd:
            self.dx = self.x_coords_domain[1] - self.x_coords_domain[0] if len(self.x_coords_domain) > 1 else 1.0
            if not np.allclose(np.diff(self.x_coords_domain), self.dx):
                print("Warning: x_coords_domain is not on a strictly regular grid. Grid-optimized WD might be inaccurate.")
        
        self.current_interactive_params = { name: config[2] for name, config in self.interactive_param_configs.items() }
        
        self._setup_plot()
        self._create_sliders()
        self.update_plot(None)

    def _normalize_ys(self, ys_values):
        s = np.sum(ys_values)
        # Ensure non-negative output before normalization
        ys_non_negative = np.maximum(ys_values, 0)
        s_non_negative = np.sum(ys_non_negative)
        return ys_non_negative / s_non_negative if s_non_negative > 1e-9 else np.zeros_like(ys_values)

    def _setup_plot(self):
        num_params = len(self.interactive_param_configs)
        fig_height = 6 + num_params * 0.5
        self.fig, (self.ax_dist, self.ax_cdf) = plt.subplots(2, 1, figsize=(10, fig_height), sharex=True)
        plt.subplots_adjust(bottom=max(0.15, 0.05 * num_params + 0.12))

        # Initial reference distribution y-values
        self.ref_ys_raw = self.reference_model_func(self.x_coords_domain, self.reference_params)
        self.ref_ys_norm = self._normalize_ys(self.ref_ys_raw)
        
        # Initial interactive distribution y-values
        interactive_ys_raw_init = self.interactive_model_func(self.x_coords_domain, self.current_interactive_params)
        interactive_ys_norm_init = self._normalize_ys(interactive_ys_raw_init)

        # Plot distributions (y-values against x_coords_domain)
        self.line_ref_dist, = self.ax_dist.plot(self.x_coords_domain, self.ref_ys_norm, lw=2, label='Reference Signal (y=f(x))')
        self.line_interactive_dist, = self.ax_dist.plot(self.x_coords_domain, interactive_ys_norm_init, lw=2, label='Interactive Signal (y=g(x))', color='orange')
        self.ax_dist.set_title('1D Signals (Normalized y-values treated as mass)')
        self.ax_dist.set_ylabel('Normalized Intensity (y)')
        self.ax_dist.legend(loc='upper right')
        self.ax_dist.grid(True)

        # Plot CDFs
        self.ref_cdf_vals = np.cumsum(self.ref_ys_norm)
        interactive_cdf_vals_init = np.cumsum(interactive_ys_norm_init)
        self.line_ref_cdf, = self.ax_cdf.plot(self.x_coords_domain, self.ref_cdf_vals, lw=2, label='CDF Reference')
        self.line_interactive_cdf, = self.ax_cdf.plot(self.x_coords_domain, interactive_cdf_vals_init, lw=2, label='CDF Interactive', color='orange')
        
        self.fill_cdf_obj = self.ax_cdf.fill_between(self.x_coords_domain, self.ref_cdf_vals, interactive_cdf_vals_init,
                                                     alpha=0.3, color='gray', step='post', label='|CDF_Ref-CDF_Int| Area')
        self.ax_cdf.set_title('Cumulative Distribution Functions (CDFs)')
        self.ax_cdf.set_xlabel('Position (x)')
        self.ax_cdf.set_ylabel('Cumulative Mass')
        self.ax_cdf.legend(loc='center right')
        self.ax_cdf.grid(True)
        self.ax_cdf.set_ylim(-0.05, 1.05)

        self.wd_text = self.ax_dist.text(0.05, 0.9, '', transform=self.ax_dist.transAxes, fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
        
        self._update_dist_ylim(self.ref_ys_norm, interactive_ys_norm_init)

    def _update_dist_ylim(self, ys1, ys2):
        min_val = min(0, np.min(ys1) if ys1.size > 0 else 0, np.min(ys2) if ys2.size > 0 else 0)
        max_val = max(0.01, 1.1 * np.max(ys1) if ys1.size > 0 else 0.01, 1.1 * np.max(ys2) if ys2.size > 0 else 0.01)
        # Ensure min_val is not greater than max_val if all ys are zero
        if min_val > max_val: min_val = max_val - 0.01 
        self.ax_dist.set_ylim(bottom=min_val, top=max_val)

    def _create_sliders(self):
        self.sliders = {}
        self.slider_axes = []
        slider_base_y = 0.025
        slider_height = 0.03
        slider_spacing = 0.015

        for i, (param_name, config) in enumerate(self.interactive_param_configs.items()):
            min_val, max_val, init_val, step_val = config
            ax_s = self.fig.add_axes([0.20, slider_base_y + i * (slider_height + slider_spacing),   0.65, slider_height], facecolor='lightgoldenrodyellow')
            self.slider_axes.append(ax_s)
            self.sliders[param_name] = Slider( ax=ax_s, label=param_name, valmin=min_val, valmax=max_val, valinit=init_val, valstep=step_val )
            self.sliders[param_name].on_changed(self.update_plot)
            
    def update_plot(self, val):
        for p_name in self.sliders:
            self.current_interactive_params[p_name] = self.sliders[p_name].val

        interactive_ys_raw = self.interactive_model_func(self.x_coords_domain, self.current_interactive_params)
        interactive_ys_norm = self._normalize_ys(interactive_ys_raw)

        self.line_interactive_dist.set_ydata(interactive_ys_norm)
        self._update_dist_ylim(self.ref_ys_norm, interactive_ys_norm)

        interactive_cdf_vals = np.cumsum(interactive_ys_norm)
        self.line_interactive_cdf.set_ydata(interactive_cdf_vals)

        if self.fill_cdf_obj: self.fill_cdf_obj.remove()
        self.fill_cdf_obj = self.ax_cdf.fill_between(self.x_coords_domain, self.ref_cdf_vals, interactive_cdf_vals, alpha=0.3, color='gray', step='post')

        if self.use_grid_optimized_wd:
            wd = wasserstein_1d_grid(self.ref_ys_norm, interactive_ys_norm, self.dx)
        else:
            # For general W1, x_coords_domain serves as the 'xs' for both distributions
            # as they are evaluated on this common domain by the model functions.
            wd = wasserstein_1d_general(self.x_coords_domain, self.ref_ys_norm,  self.x_coords_domain, interactive_ys_norm)
        
        self.wd_text.set_text(f'Wasserstein Distance (W1): {wd:.4f}')
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()