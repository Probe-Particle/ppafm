r'''

## USER

lets try different version of the generator of boundary points. 
1) we take COG of the molecule, and generate angular sampling by arc-lengh (angle multiplied by radius), the radius is given by the furthest atom in that arc plus some margin. For that probably to make it efficintly it is good to sort points by angle from COG, if you understand what I mean?
2) when we have this first layer we can simply extend it by few steps from the COG (alays ading constant radial offest
3) in the first iteration we can make staggered offest (where we make point in the center of each arc segment (in between angles of ppint in the 1st layer), and ofset them by half of the radial offset, hope this is understandable.

---

## Gemini

This is a **Radial "Onion-Peeling" Approach**. It is very computationally efficient because it reduces the 2D problem to a 1D problem (angles).

### The Concept

1.  **Center of Geometry (COG):** We move the origin to the center of the molecule.
2.  **Envelope Detection:** instead of complex geometry, we scan the molecule radially. We divide the $360^{\circ}$ circle into bins. In each bin, we find the **furthest atom**. This creates a "Max Radius Profile" ($R(\theta)$).
3.  **Smoothing:** We smooth this profile so the generated grid doesn't jump wildly between atoms.
4.  **Layer Generation:**
    *   We generate points in concentric rings (layers) starting from $R(\theta) + \text{margin}$.
    *   **Arc-Length Sampling:** The angular step $\Delta \theta$ is calculated as $S / r$, where $S$ is your target spacing. This ensures points are evenly spaced even as the circle gets bigger.
    *   **Staggering:** Every second layer is rotated by half an angular step to create a "brick wall" or hexagonal packing pattern.

### Why this fits your request

1.  **COG & Angular Sorting:** Uses `get_max_radius_profile` to effectively sort the geometry by angle relative to the COG.
2.  **Furthest Atom + Margin:** It specifically scans each angular bin for the maximum radius (`max_r`) and adds the `margin` to establish the baseline for the first layer.
3.  **Arc-Length Sampling:** The loop `d_theta = spacing / r_point` guarantees that points are `spacing` distance apart along the circumference, regardless of how far out they are.
4.  **Staggered Offset:** It detects odd layers (`layer_idx % 2 == 1`) and shifts the `theta` by half the angular step of that layer, creating the requested interleaved pattern.

This generates a very organic "force-field" style mesh that conforms perfectly to the overall shape (convex hull) of the molecule.

'''

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

PRESET_MOLECULES = [
    "OHO-h_1_points_clean.txt",
    "HN-hh_point_info.txt",
    "PTCDA_point_info.txt",
]

# ==========================================
# 1. IO & Math Utilities
# ==========================================

def load_points_file(path: str) -> Tuple[List[str], np.ndarray]:
    """Robustly loads points from various text formats."""
    types = []
    coords = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
    start = 1 if len(lines) > 0 and ("type" in lines[0] or "index" in lines[0]) else 0
    
    for line in lines[start:]:
        parts = line.strip().replace(',', ' ').replace('[', ' ').replace(']', ' ').split()
        if len(parts) < 3: continue
        
        # Heuristics to find x, y
        try:
            # Format: index type x y ...
            if parts[0].isdigit() and len(parts) >= 4:
                types.append(parts[1])
                coords.append((float(parts[2]), float(parts[3])))
            # Format: type x y ...
            else:
                types.append(parts[0])
                coords.append((float(parts[1]), float(parts[2])))
        except ValueError:
            continue
            
    return types, np.array(coords)

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# ==========================================
# 2. Envelope & Grid Generators
# ==========================================

def get_max_radius_profile(coords: np.ndarray, n_bins=72) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scans the molecule angularly to find the 'skyline' (furthest atom) 
    in each direction.
    """
    if len(coords) == 0:
        return np.linspace(-np.pi, np.pi, n_bins), np.zeros(n_bins)

    r, theta = cart2pol(coords[:, 0], coords[:, 1])
    
    # Define bins (-pi to pi)
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Digitizing
    inds = np.digitize(theta, bins) - 1
    
    max_r = np.zeros(n_bins)
    
    for i in range(n_bins):
        # Find atoms in this angular wedge
        radii_in_bin = r[inds == i]
        if len(radii_in_bin) > 0:
            max_r[i] = np.max(radii_in_bin)
        else:
            # If gap, temporarily set to 0, fix later
            max_r[i] = 0.0
            
    # Gap filling: Linear interpolation for empty wedges
    # (Fixes "U" shapes or gaps between distant atoms)
    zero_mask = max_r == 0
    if np.any(zero_mask) and not np.all(zero_mask):
        x_vals = np.arange(n_bins)
        max_r[zero_mask] = np.interp(x_vals[zero_mask], x_vals[~zero_mask], max_r[~zero_mask])
    elif np.all(zero_mask):
        max_r[:] = 1.0 # Fallback

    # Smoothing: Moving average to prevent jagged grid lines
    # This wraps around the array (periodic boundary)
    kernel_radius = 2  # results in 5-point moving average
    if kernel_radius > 0:
        smooth = np.zeros_like(max_r)
        count = 0
        for shift in range(-kernel_radius, kernel_radius + 1):
            smooth += np.roll(max_r, shift)
            count += 1
        max_r = smooth / count

    return bin_centers, max_r

def generate_radial_grid(
    coords: np.ndarray,
    spacing: float,       # Arc-length spacing (distance between points in ring)
    radial_step: float,   # Distance between rings
    margin: float,        # Start distance from furthest atom
    n_layers: int,
    stagger: bool = True
) -> np.ndarray:
    
    # 1. Center coordinates to COG
    cog = np.mean(coords, axis=0)
    centered_coords = coords - cog
    
    # 2. Get the Envelope (The "Boundary")
    # We use high resolution bins for lookup
    profile_theta, profile_r = get_max_radius_profile(centered_coords, n_bins=180)
    profile_theta_wrapped = np.concatenate([profile_theta, profile_theta[:1] + 2 * np.pi])
    profile_r_wrapped = np.concatenate([profile_r, profile_r[:1]])
    
    grid_points = []
    
    # 3. Generate Layers (Onion Peeling)
    for layer_idx in range(n_layers):
        
        points_in_layer = []
        current_theta = -np.pi
        
        # Determine base radius for this layer relative to envelope
        # We assume the envelope expands linearly with layers
        current_radial_offset = margin + (layer_idx * radial_step)
        
        # Stagger logic: Shift starting angle
        # If staggered, we shift the 2nd, 4th, etc layer by half an arc-step
        is_staggered_layer = stagger and (layer_idx % 2 == 1)
        
        while current_theta < np.pi:
            # Interpolate the molecule's radius at this specific angle
            # (We use interpolation so we aren't limited to bin resolution)
            mol_r = np.interp(current_theta, profile_theta_wrapped, profile_r_wrapped)
            
            # The radius of the grid point
            r_point = mol_r + current_radial_offset
            
            # Add point
            points_in_layer.append((r_point, current_theta))
            
            # Calculate next angle
            # Arc Length Formula: s = r * theta  ->  theta = s / r
            d_theta = spacing / r_point
            
            current_theta += d_theta
        
        # Convert layer to Cartesian
        pts_pol = np.array(points_in_layer)
        if len(pts_pol) > 0:
            
            # Apply Angular Shift for Staggering
            if is_staggered_layer:
                # We calculate average d_theta for this layer to estimate half-step
                avg_r = np.mean(pts_pol[:,0])
                shift = (spacing / avg_r) * 0.5
                pts_pol[:, 1] += shift
            
            # Transform back to Cartesian + add COG offset
            xs, ys = pol2cart(pts_pol[:,0], pts_pol[:,1])
            layer_cart = np.column_stack([xs, ys]) + cog
            grid_points.append(layer_cart)

    if not grid_points:
        return np.empty((0, 2))
        
    return np.vstack(grid_points)

# ==========================================
# 3. Main Processing
# ==========================================

def process_molecule(fname: str, args):
    path = os.path.join(args.points_dir, fname)
    types, coords = load_points_file(path)
    
    if len(coords) == 0: return None

    # Generate the Radial Grid
    # Note: radial_step defaults to spacing * sin(60) ~ 0.866 * spacing for perfect hex, 
    # or just spacing for square-ish radial.
    radial_step = args.spacing * 0.866
    
    grid_pts = generate_radial_grid(
        coords, 
        spacing=args.spacing, 
        radial_step=radial_step, 
        margin=args.margin, 
        n_layers=args.layers,
        stagger=True
    )

    # Optional: Hard Cutoff filter (Double check)
    # Since we generate based on "Max Radius", some points might still 
    # sneak into concave bays. Let's do a quick distance check.
    from scipy.spatial.distance import cdist
    dists = cdist(grid_pts, coords).min(axis=1)
    grid_pts = grid_pts[dists >= args.margin * 0.9] # 0.9 tolerance

    all_coords = np.vstack([coords, grid_pts])
    all_types = types + ['grid'] * len(grid_pts)
    
    return coords, grid_pts, all_coords, all_types

def plot_result(ax, orig, grid, title):
    if len(grid):
        ax.scatter(grid[:,0], grid[:,1], c='tab:orange', s=5, label='Radial Grid')
    ax.scatter(orig[:,0], orig[:,1], c='black', s=20, label='Molecule')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--points-dir", default="data_Mithun/points")
    p.add_argument("-o", "--out-dir",    default="data_Mithun/points_rad")
    p.add_argument("-s", "--spacing",    type=float, default=1.5, help="Arc length spacing")
    p.add_argument("-m", "--margin",     type=float, default=0.7, help="Initial distance from atoms")
    p.add_argument("-l", "--layers",     type=int,   default=5,   help="Number of radial layers")
    p.add_argument("--show",             type=int,   default=1)
    
    args = p.parse_args()
    
    # Demo data generation
    if not os.path.exists(args.points_dir):
        os.makedirs(args.points_dir, exist_ok=True)
        # Ellipse
        t = np.linspace(0, 2*np.pi, 20)
        pts = np.column_stack([np.cos(t)*4, np.sin(t)*2])
        with open(os.path.join(args.points_dir, "ellipse.txt"), "w") as f:
            f.write("type x y\n")
            for x,y in pts: f.write(f"C {x} {y}\n")

    if PRESET_MOLECULES:
        files = PRESET_MOLECULES
    else:
        files = [f for f in os.listdir(args.points_dir) if f.endswith(".txt")]
    
    ncols = min(3, len(files))
    nrows = int(np.ceil(len(files)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.atleast_1d(axes).ravel()
    
    for i, f in enumerate(files):
        res = process_molecule(f, args)
        if res:
            orig, grid, all_p, all_t = res
            plot_result(axes[i], orig, grid, f)
            
            # Save
            out = os.path.join(args.out_dir, f.replace(".txt", "_rad.txt"))
            os.makedirs(args.out_dir, exist_ok=True)
            with open(out, "w") as fo:
                fo.write("type x y\n")
                for t, (x,y) in zip(all_t, all_p):
                    fo.write(f"{t} {x:.5f} {y:.5f}\n")

    plt.tight_layout()
    if args.show: plt.show()