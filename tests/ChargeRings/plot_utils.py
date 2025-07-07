import numpy as np
import matplotlib.pyplot as plt

diverting_cmaps = set(['PiYG','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'berlin', 'managua', 'vanimo', 'vanimo', 'PuRdR-w-BuGn', 'BuGnR-w-PuRd'])

def plot_imshow( ax, data, title=None, extent=None, spos=None, cmap=None, vmin=None, vmax=None, xlabel="x [Å]", ylabel="y [Å]", bDiverging=False, bGrid=False, scV=1.0, **kwargs):
    if cmap is not None:
        if isinstance(cmap, str):
            cmap_ = cmap.split("_")[0]
            if (cmap_ in diverting_cmaps) or (cmap_+"_r" in diverting_cmaps):
                bDiverging = True
    if bDiverging:    
        #print("plot_imshow() bDiverging", bDiverging, "  cmap: ", cmap)
        if vmin is None or vmax is None:
            vmax = np.max(np.abs(data))*scV
            vmin = -vmax
    ax.clear()
    ax.imshow(data, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, **kwargs)
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