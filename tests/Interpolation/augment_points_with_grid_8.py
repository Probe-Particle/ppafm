import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# PRESET_MOLECULES = [
#     "OHO-h_1_points_clean.txt",
#     "HN-hh_point_info.txt",
#     "PTCDA_point_info.txt",
# ]

POINT_INFO_RE = re.compile(r"^\s*(\d+)?\s*([A-Za-z0-9_]+)\s*\[([^\]]+)\]")
NON_ATOM_TYPES = {"center", "bond", "cp", "grid"}

# ==========================================
# 1. The Parabolic Envelope Engine
# ==========================================

def get_smooth_parabolic_envelope(
    coords: np.ndarray, 
    margin: float, 
    stiffness: float = 0.5, 
    n_bins: int = 1080  # High res for accurate lookups
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a smooth envelope by treating each atom as a parabola source.
    """
    if len(coords) == 0:
        return np.linspace(-np.pi, np.pi, n_bins), np.zeros(n_bins), np.zeros(2)

    # 1. Center to COG
    cog = np.mean(coords, axis=0)
    centered = coords - cog
    
    # 2. Convert atoms to Polar
    r_atoms = np.sqrt(np.sum(centered**2, axis=1))
    theta_atoms = np.arctan2(centered[:, 1], centered[:, 0])
    
    # 3. Create Angular Grid
    grid_angles = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    
    # 4. Vectorized Parabola Calculation
    d_theta = np.abs(theta_atoms[:, None] - grid_angles[None, :])
    d_theta = np.minimum(d_theta, 2 * np.pi - d_theta)
    
    arc_dist = r_atoms[:, None] * d_theta
    peak_heights = r_atoms[:, None] + margin
    parabolas = peak_heights - stiffness * (arc_dist**2)
    
    # 5. The Envelope is the MAX of all parabolas
    envelope_r = np.max(parabolas, axis=0)
    envelope_r = np.maximum(envelope_r, margin * 0.5)
    envelope_r = np.nan_to_num(envelope_r, nan=margin)
    
    return grid_angles, envelope_r, cog

# ==========================================
# 2. Layer Generation (Fixed N + Staggering)
# ==========================================

def get_circular_midpoints(angles: np.ndarray) -> np.ndarray:
    """
    Calculates the angular midpoints between consecutive angles in a sorted array,
    handling the wrap-around from the last point to the first.
    Returns array of same length as input.
    """
    # Use complex numbers to handle circular mean easily without boundary checks
    # z = exp(i*theta)
    z = np.exp(1j * angles)
    # Roll to pair i with i+1
    z_next = np.roll(z, -1)
    
    # Mean vector
    z_mean = z + z_next
    
    # Angle of mean vector
    mid_angles = np.angle(z_mean)
    
    return mid_angles

def generate_parabolic_grid(
    coords: np.ndarray,
    spacing: float,
    margin: float,
    radial_steps: List[float],
    stiffness: float = 0.1,
    staggered: bool = False
) -> np.ndarray:
    
    # 1. Get high-resolution envelope for lookups
    env_theta_grid, env_r_grid, cog = get_smooth_parabolic_envelope(
        coords, margin=margin, stiffness=stiffness, n_bins=1440
    )
    
    # 2. Generate Layer 0 (The Base Layer) using Arc-Length Sampling
    # We calculate the exact shape of Layer 0 to determine the number of points N
    base_step = radial_steps[0]
    layer0_r_grid = np.maximum(env_r_grid + base_step, 1e-6)
    
    # Close loop for integration
    theta_closed = np.concatenate([env_theta_grid, [env_theta_grid[0] + 2*np.pi]])
    r_closed = np.concatenate([layer0_r_grid, [layer0_r_grid[0]]])
    
    x_closed = r_closed * np.cos(theta_closed)
    y_closed = r_closed * np.sin(theta_closed)
    
    # Integrate arc length
    dists = np.sqrt(np.diff(x_closed)**2 + np.diff(y_closed)**2)
    cum_dist = np.concatenate([[0], np.cumsum(dists)])
    total_len = cum_dist[-1]
    
    # Determine N (Fixed for all layers)
    n_points = max(3, int(np.round(total_len / spacing)))
    
    # Get Layer 0 Angles
    target_dists = np.linspace(0, total_len, n_points, endpoint=False)
    current_angles = np.interp(target_dists, cum_dist, theta_closed)
    
    # Normalize angles to -pi, pi
    current_angles = (current_angles + np.pi) % (2 * np.pi) - np.pi
    
    all_layers_pts = []

    # 3. Iterate over steps
    for i, step in enumerate(radial_steps):
        
        # Staggering Logic:
        # If staggered is ON and this is not the very first iteration loop (conceptually),
        # we average the previous angles.
        # Note: We do this for every step after the first one in the list.
        if staggered and i > 0:
            current_angles = get_circular_midpoints(current_angles)

        # Lookup Envelope Radius for these specific angles
        # We look up the *Base* envelope radius at these angles
        # This preserves the shape
        base_r = np.interp(current_angles, env_theta_grid, env_r_grid, period=2*np.pi)
        
        # Add the radial offset
        final_r = np.maximum(base_r + step, 1e-6)
        
        # Convert to Cartesian
        lx = final_r * np.cos(current_angles)
        ly = final_r * np.sin(current_angles)
        
        all_layers_pts.append(np.column_stack([lx, ly]))

    if not all_layers_pts:
        return np.empty((0, 2))

    return np.vstack(all_layers_pts) + cog

# ==========================================
# 3. Main & Plotting
# ==========================================

def load_points_file(path):
    types: List[str] = []
    coords: List[Tuple[float, float]] = []
    if not os.path.exists(path):
        return types, np.array(coords)

    with open(path) as f:
        lines = f.readlines()

    start = 1 if lines and ("type" in lines[0] or "index" in lines[0]) else 0

    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        parsed = None
        try:
            if len(parts) >= 4 and parts[0].isdigit():
                parsed = (parts[1], float(parts[2]), float(parts[3]))
            elif len(parts) >= 3:
                parsed = (parts[0], float(parts[1]), float(parts[2]))
        except ValueError:
            parsed = None

        if parsed is None:
            m = POINT_INFO_RE.match(line)
            if m:
                coord_tokens = m.group(3).replace(",", " ").split()
                if len(coord_tokens) >= 2:
                    try:
                        parsed = (m.group(2), float(coord_tokens[0]), float(coord_tokens[1]))
                    except ValueError:
                        parsed = None

        if parsed is None:
            continue

        typ, x, y = parsed
        types.append(typ)
        coords.append((x, y))

    return types, np.array(coords, dtype=float)


def extract_atom_coords(types: List[str], coords: np.ndarray) -> Tuple[np.ndarray, int]:
    if coords.size == 0:
        return coords, 0
    atom_coords = []
    for typ, coord in zip(types, coords):
        if typ.lower() in NON_ATOM_TYPES:
            continue
        atom_coords.append(coord)
    if not atom_coords:
        return coords, len(coords)
    atom_arr = np.asarray(atom_coords, dtype=float)
    return atom_arr, len(atom_arr)


def compute_bboxes(coords: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns ((orig_min, orig_max), (outer_min, outer_max))."""
    if coords.size == 0:
        zero = (np.zeros(2), np.zeros(2))
        return zero, zero
    orig_min = coords.min(axis=0)
    orig_max = coords.max(axis=0)
    outer_min = orig_min - margin
    outer_max = orig_max + margin
    return (orig_min, orig_max), (outer_min, outer_max)


def clip_to_bbox(points: np.ndarray, bbox: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if points.size == 0:
        return points
    (bmin, bmax) = bbox
    mask = (
        (points[:,0] >= bmin[0]) & (points[:,0] <= bmax[0]) &
        (points[:,1] >= bmin[1]) & (points[:,1] <= bmax[1])
    )
    return points[mask]


def draw_bbox(ax, bbox, **kwargs):
    (bmin, bmax) = bbox
    rect = plt.Rectangle(
        (bmin[0], bmin[1]),
        bmax[0] - bmin[0],
        bmax[1] - bmin[1],
        fill=False,
        **kwargs,
    )
    ax.add_patch(rect)


def save_points_file(path: str, coords: np.ndarray, types: List[str]) -> None:
    if coords.size == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("type x y\n")
        for t, (x, y) in zip(types, coords):
            f.write(f"{t} {x:.5f} {y:.5f}\n")


def plot_snapshot(ax, coords, atom_coords, grid, env_x, env_y, orig_bbox, outer_bbox, label_new, title):
    ax.plot(env_x, env_y, color='red', linestyle='--', linewidth=0.5, label='Envelope')
    draw_bbox(ax, outer_bbox, edgecolor='tab:blue', linestyle=':', lw=1.1, alpha=0.9)
    draw_bbox(ax, orig_bbox, edgecolor='green', linestyle='-.', lw=0.8, alpha=0.8)

    if len(coords):
        ax.scatter(coords[:,0], coords[:,1], c='0.7', s=8, zorder=3, label='All points')
    if atom_coords.size:
        ax.scatter(atom_coords[:,0], atom_coords[:,1], c='black', s=35, zorder=4, label='Atoms')
    if len(grid):
        ax.scatter(grid[:,0], grid[:,1], c='tab:blue', s=12, alpha=0.7, label='Grid')
        if label_new:
            for idx, (x, y) in enumerate(grid):
                ax.text(x, y, str(idx), fontsize=6, color='k', alpha=1.)

    ax.set_title(title)
    ax.set_aspect('equal')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--points-dir", default="data_Mithun/points")
    p.add_argument("-o", "--out-dir",    default="data_Mithun/points_outer")
    p.add_argument("-s", "--spacing",    type=float,   default=1.0, help="Grid point spacing (defined at layer 0)")
    p.add_argument("-m", "--margin",     type=float,   default=0.7, help="Base height of the envelope above atoms")
    p.add_argument("-M", "--outer-margin", type=float, default=4.0, help="Additional padding for plotting/clip bbox")
    p.add_argument("-k", "--stiffness",  type=float, default=0.3,  help="Parabola decay.")
    p.add_argument("-l", "--layers",     type=int,   default=5)
    p.add_argument("--radial-steps",     type=str,   default="0.0,1.0,2.5,4.0,6.0", help="Comma-separated radial offsets")
    p.add_argument("--staggered",        type=int,  default=1, help="If set, every subsequent layer averages the angles of the previous layer.")
    p.add_argument("--label-new",        type=int,  default=1, help="If 1, label generated grid points")
    p.add_argument("--save-png",         type=int,  default=1, help="If 1, save per-molecule PNGs")
    p.add_argument("--fig-dir",          type=str,  default="data_Mithun/plots_png", help="Directory for PNG outputs")
    p.add_argument("--save-new",         type=int,  default=1, help="If 1, save only newly generated points to txt")
    p.add_argument("--new-dir",          type=str,  default="data_Mithun/points_new", help="Directory for new-point txt files")
    p.add_argument("--save-combined",    type=int,  default=0, help="If 1, save combined original+grid points")
    p.add_argument("--combined-dir",     type=str,  default="data_Mithun/points_aug_combined", help="Directory for combined txt files")
    p.add_argument("--show",             type=int,   default=1)
    args = p.parse_args()

    # Generate test data if needed
    if not os.path.exists(args.points_dir):
        os.makedirs(args.points_dir, exist_ok=True)
        pts = [[0,0], [2,2], [2,-2], [4,4], [4,-4]]
        with open(os.path.join(args.points_dir, "v_shape.txt"), "w") as f:
            f.write("type x y\n")
            for x,y in pts: f.write(f"C {x} {y}\n")

    radial_steps = [float(s) for s in args.radial_steps.split(",") if s.strip()]
    if len(radial_steps) != args.layers:
        if len(radial_steps) < args.layers:
            radial_steps += [radial_steps[-1] + (radial_steps[-1]-radial_steps[-2]) * (i+1) for i in range(args.layers - len(radial_steps))]
        else:
            radial_steps = radial_steps[:args.layers]
    args.radial_steps = radial_steps

    preset_files = globals().get("PRESET_MOLECULES", [])
    if preset_files:
        files = preset_files
    else:
        files = [f for f in os.listdir(args.points_dir) if f.endswith(".txt")]
    
    ncols = min(3, len(files))
    nrows = int(np.ceil(len(files)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, fname in enumerate(files):
        types, coords = load_points_file(os.path.join(args.points_dir, fname))
        if len(coords) == 0:
            continue
        atom_coords, atom_count = extract_atom_coords(types, coords)
        if atom_coords.size == 0:
            atom_coords = coords
            atom_count = len(coords)

        (orig_bbox, outer_bbox) = compute_bboxes(coords, args.outer_margin)

        print(f"[DEBUG] {fname}: original={len(coords)} atoms_used={atom_count}", end="")

        # Generate Grid
        grid = generate_parabolic_grid(
            atom_coords, 
            spacing=args.spacing, 
            margin=args.margin, 
            radial_steps=args.radial_steps,
            stiffness=args.stiffness,
            staggered=args.staggered
        )
        grid = clip_to_bbox(grid, outer_bbox)
        print(f" new={len(grid)} total={len(coords)+len(grid)}")
        
        ax = axes[i]
        
        # Visual Debug: Envelope
        env_theta, env_r, cog = get_smooth_parabolic_envelope(atom_coords, args.margin, args.stiffness)
        env_theta = np.append(env_theta, env_theta[0])
        env_r = np.append(env_r, env_r[0])
        env_x = cog[0] + env_r * np.cos(env_theta)
        env_y = cog[1] + env_r * np.sin(env_theta)
        
        plot_snapshot(
            ax, coords, atom_coords, grid, env_x, env_y,
            orig_bbox, outer_bbox, bool(args.label_new),
            f"{fname}\norig={len(coords)} grid={len(grid)}",
        )
        
        out_path = os.path.join(args.out_dir, fname.replace(".txt", "_smooth.txt"))
        os.makedirs(args.out_dir, exist_ok=True)
        all_pts = np.vstack([coords, grid]) if len(grid) else coords
        all_types = types + ['grid']*len(grid)
        save_points_file(out_path, all_pts, all_types)

        if args.save_new and len(grid):
            new_path = os.path.join(args.new_dir, fname.replace(".txt", "_grid_only.txt"))
            save_points_file(new_path, grid, ['grid']*len(grid))

        if args.save_combined:
            comb_path = os.path.join(args.combined_dir, fname.replace(".txt", "_combined.txt"))
            save_points_file(comb_path, all_pts, all_types)

        if args.save_png:
            os.makedirs(args.fig_dir, exist_ok=True)
            fig_single, ax_single = plt.subplots(figsize=(6,6))
            plot_snapshot(
                ax_single, coords, atom_coords, grid, env_x, env_y,
                orig_bbox, outer_bbox, bool(args.label_new),
                f"{fname}\norig={len(coords)} grid={len(grid)}",
            )
            fig_single.savefig(os.path.join(args.fig_dir, fname.replace(".txt", ".png")), dpi=200)
            plt.close(fig_single)

    plt.tight_layout()
    if args.show:
        plt.show()