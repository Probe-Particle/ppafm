import numpy as np
import matplotlib.pyplot as plt

diverting_cmaps = set(['PiYG','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'berlin', 'managua', 'vanimo', 'vanimo', 'PuRdR-w-BuGn', 'BuGnR-w-PuRd'])

def plot_imshow( ax, data, title=None, extent=None, spos=None, cmap=None, vmin=None, vmax=None, xlabel="x [Å]", ylabel="y [Å]", bDiverging=False, bGrid=False, scV=1.0, **kwargs):
    if cmap is not None:
        if isinstance(cmap, str):
            cmap_ = cmap.split("_")[0]
            if (cmap_ in diverting_cmaps) or (cmap_+"_r" in diverting_cmaps):
                bDiverging = True
    if vmin is None or vmax is None:
        if bDiverging:    
            # If data crosses zero, use symmetric range
            if np.min(data) < 0 and np.max(data) > 0:
                vmax_abs = np.max(np.abs(data))*scV
                if vmax_abs == 0.0: vmax_abs = 1e-9 # Avoid vmax=0.0 causing vmin=vmax error
                vmin_final = -vmax_abs
                vmax_final = vmax_abs
            else:
                # If data is entirely positive or entirely negative, use actual min/max
                vmin_final = np.min(data)
                vmax_final = np.max(data)
                if vmin_final == vmax_final: # Handle flat data
                    if vmin_final == 0.0: # If data is all zeros
                        vmin_final = -1e-9
                        vmax_final = 1e-9
                    else: # If data is a non-zero constant
                        vmin_final = vmin_final * 0.9
                        vmax_final = vmax_final * 1.1
        else:
            # For non-diverging, calculate from data if not provided
            vmin_final = np.min(data)
            vmax_final = np.max(data)
            if vmin_final == vmax_final: # Handle flat data
                if vmin_final == 0.0: # If data is all zeros
                    vmin_final = -1e-9
                    vmax_final = 1e-9
                else: # If data is a non-zero constant
                    vmin_final = vmin_final * 0.9
                    vmax_final = vmax_final * 1.1
    else:
        # If vmin/vmax are provided, ensure vmin <= vmax
        if vmin > vmax:
            vmin_final, vmax_final = vmax, vmin # Swap if inverted
        else:
            vmin_final = vmin
            vmax_final = vmax

    # Debugging print statements
    print(f"plot_imshow Debug: data.shape={data.shape}, data.min={np.min(data)}, data.max={np.max(data)}")
    print(f"plot_imshow Debug: vmin_final={vmin_final}, vmax_final={vmax_final}")
    if vmin_final > vmax_final:
        print("plot_imshow ERROR: vmin_final is greater than vmax_final!")

    ax.clear()
    ax.imshow(data, extent=extent, cmap=cmap, origin='lower', vmin=vmin_final, vmax=vmax_final, **kwargs)
    if spos is not None:
        for i in range(spos.shape[0]):
            ax.plot(spos[i,0], spos[i,1], 'ro')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if bGrid:
        ax.grid()
    return ax