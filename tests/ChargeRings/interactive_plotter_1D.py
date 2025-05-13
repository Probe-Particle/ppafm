# interactive_plotter.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class InteractivePlotter1D:
    def __init__(self,
                 x_coords_domain: np.ndarray, # The common x-coordinates for model evaluation & plotting
                 reference_model_func,       # func(x_coords, params_dict) -> y_values
                 reference_params: dict,
                 interactive_model_func,     # func(x_coords, params_dict) -> y_values
                 interactive_param_configs: dict, # name: (min, max, init, step)
                 metric_callback=None,       # Optional: func(ref_ys, interactive_ys, x_coords) -> (metric_value, metric_text)
                 dist_title='1D Signals (Normalized y-values)',
                 cdf_title='Cumulative Distribution Functions (CDFs)'
                ):
        self.x_coords_domain = x_coords_domain
        self.reference_model_func = reference_model_func
        self.reference_params = reference_params
        self.interactive_model_func = interactive_model_func
        self.interactive_param_configs = interactive_param_configs
        self.metric_callback = metric_callback
        self.dist_title = dist_title
        self.cdf_title = cdf_title
        
        self.current_interactive_params = { name: config[2] for name, config in self.interactive_param_configs.items() }
        
        self._setup_plot()
        self._create_sliders()
        self.update_plot(None)

    def _normalize_ys(self, ys_values):
        s = np.sum(ys_values)
        ys_non_negative = np.maximum(ys_values, 0)
        s_non_negative = np.sum(ys_non_negative)
        return ys_non_negative / s_non_negative if s_non_negative > 1e-9 else np.zeros_like(ys_values)

    def _setup_plot(self):
        num_params = len(self.interactive_param_configs)
        fig_height = 6 + num_params * 0.5
        self.fig, (self.ax_dist, self.ax_cdf) = plt.subplots(2, 1, figsize=(10, fig_height), sharex=True)
        plt.subplots_adjust(bottom=max(0.15, 0.05 * num_params + 0.12))

        self.ref_ys_raw = self.reference_model_func(self.x_coords_domain, self.reference_params)
        self.ref_ys_norm = self._normalize_ys(self.ref_ys_raw)
        
        interactive_ys_raw_init = self.interactive_model_func(self.x_coords_domain, self.current_interactive_params)
        interactive_ys_norm_init = self._normalize_ys(interactive_ys_raw_init)

        self.line_ref_dist, = self.ax_dist.plot(self.x_coords_domain, self.ref_ys_norm, lw=2, label='Reference Signal')
        self.line_interactive_dist, = self.ax_dist.plot(self.x_coords_domain, interactive_ys_norm_init, lw=2, label='Interactive Signal', color='orange')
        self.ax_dist.set_title(self.dist_title)
        self.ax_dist.set_ylabel('Normalized Intensity (y)')
        self.ax_dist.legend(loc='upper right')
        self.ax_dist.grid(True)

        self.ref_cdf_vals = np.cumsum(self.ref_ys_norm)
        interactive_cdf_vals_init = np.cumsum(interactive_ys_norm_init)
        self.line_ref_cdf, = self.ax_cdf.plot(self.x_coords_domain, self.ref_cdf_vals, lw=2, label='CDF Reference')
        self.line_interactive_cdf, = self.ax_cdf.plot(self.x_coords_domain, interactive_cdf_vals_init, lw=2, label='CDF Interactive', color='orange')
        self.fill_cdf_obj = self.ax_cdf.fill_between(self.x_coords_domain, self.ref_cdf_vals, interactive_cdf_vals_init,  alpha=0.3, color='gray', step='post', label='|CDF_Ref-CDF_Int| Area')
        self.ax_cdf.set_title(self.cdf_title)
        self.ax_cdf.set_xlabel('Position (x)')
        self.ax_cdf.set_ylabel('Cumulative Mass')
        self.ax_cdf.legend(loc='center right')
        self.ax_cdf.grid(True)
        self.ax_cdf.set_ylim(-0.05, 1.05)

        self.metric_text = self.ax_dist.text(0.05, 0.9, '', transform=self.ax_dist.transAxes, fontsize=12,  bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
        
        self._update_dist_ylim(self.ref_ys_norm, interactive_ys_norm_init)

    def _update_dist_ylim(self, ys1, ys2):
        min_val = min(0, np.min(ys1) if ys1.size > 0 else 0, np.min(ys2) if ys2.size > 0 else 0)
        max_val = max(0.01, 1.1 * np.max(ys1) if ys1.size > 0 else 0.01, 1.1 * np.max(ys2) if ys2.size > 0 else 0.01)
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

        if self.metric_callback:
            metric_value, metric_text = self.metric_callback(self.ref_ys_norm, interactive_ys_norm, self.x_coords_domain)
            self.metric_text.set_text(metric_text)
        
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()