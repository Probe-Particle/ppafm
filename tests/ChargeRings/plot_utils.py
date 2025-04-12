import numpy as np
import matplotlib.pyplot as plt

def plot_imshow( ax, data, title=None, extent=None, spos=None, cmap=None, vmin=None, vmax=None, **kwargs):
    if cmap is 'bwr':
        vmax = np.max(np.abs(data))
        vmin = -vmax
    ax.clear()
    ax.imshow(data, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, **kwargs)
    if spos is not None:
        for i in range(spos.shape[0]):
            ax.plot(spos[i,0], spos[i,1], 'ro')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x [Å]")
    ax.set_ylabel("y [Å]")
    ax.grid()