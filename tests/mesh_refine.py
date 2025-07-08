import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------------------------------------
#  Basic triangular grid generation (regular 2-D lattice)
# -----------------------------------------------------------

def generate_equilateral_triangle_grid(nx: int, ny: int, L=1.0):
    """Generate equilateral triangular lattice (regular hexagonal tiling).

    Parameters
    ----------
    nx, ny : int
        Number of primitive lattice steps along *a1* and *a2* directions.

    Returns
    -------
    vertices : (N,2) ndarray
    triangles : (M,3) ndarray of int
    """
    # lattice vectors under 120° (mirrored from 60°)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([-0.5, np.sqrt(3)/2])

    idx_i, idx_j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    idx_i = idx_i.ravel()
    idx_j = idx_j.ravel()
    vertices = np.outer(idx_i, a1)*L + np.outer(idx_j, a2)*L

    def vid(ix, iy):
        return iy * (nx + 1) + ix

    tris = []
    for iy in range(ny):
        for ix in range(nx):
            v00 = vid(ix, iy)
            v10 = vid(ix + 1, iy)
            v01 = vid(ix, iy + 1)
            v11 = vid(ix + 1, iy + 1)
            # two equilateral triangles per parallelogram cell
            tris.append([v00, v10, v11])  # lower
            tris.append([v00, v11, v01])  # upper
    return vertices.astype(float), np.asarray(tris, dtype=int)

# -----------------------------------------------------------
#  Geometry helpers
# -----------------------------------------------------------

def barycentric_coords(p, a, b, c):
    """Barycentric coordinates of point *p* w.r.t. triangle (a,b,c)."""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

# -----------------------------------------------------------
#  Mesh refinement primitives
# -----------------------------------------------------------

def build_edge_map_with_opposites(triangles):
    """Return mapping edge -> (opp1, opp2).

    For each undirected edge (i,j) present in exactly two adjacent triangles
    (i,j,k) and (j,i,l), store the *opposite* vertices k and l.  If an edge is
    on the boundary (only one triangle), the second opposite vertex is None.
    """
    edge_map = {}
    for tidx, (i, j, k) in enumerate(triangles):
        edges = [(i, j, k), (j, k, i), (k, i, j)]
        for a, b, opp in edges:
            key = tuple(sorted((a, b)))
            if key in edge_map:
                edge_map[key].append(opp)
            else:
                edge_map[key] = [opp]
    # convert lists to length-2 tuples, pad with None if boundary
    for key, lst in edge_map.items():
        if len(lst) == 1:
            lst.append(None)
        edge_map[key] = tuple(lst[:2])
    return edge_map

def flip_long_edges(vertices, triangles):
    """Improve mesh by flipping edges that are longer than the diagonal.

    Parameters
    ----------
    vertices : (N,2) ndarray
    triangles : (M,3) ndarray (int)

    Returns
    -------
    triangles_new : (M,3) ndarray (int)
    num_flipped : int
    """
    triangles = triangles.tolist()
    verts = np.asarray(vertices)

    edge_map = build_edge_map_with_opposites(triangles)

    flipped = 0
    for (ia, ib), (ic, id_) in edge_map.items():
        if id_ is None:
            continue  # boundary edge
        # compute squared lengths to avoid sqrt
        len_ab2 = np.sum((verts[ia] - verts[ib]) ** 2)
        len_cd2 = np.sum((verts[ic] - verts[id_] ) ** 2)
        
        #print(f"Edge {ia}-{ib} length: {np.sqrt(len_ab2):.3f}")
        #print(f"Diagonal {ic}-{id_} length: {np.sqrt(len_cd2):.3f}")
        
        if len_cd2 < len_ab2 * 0.99:  # strict inequality with small tolerance
            #print(f"  -> Should flip (diagonal is shorter)")
            # find indices of the two triangles sharing this edge
            # triangles are (ia, ib, ic) and (ib, ia, id_)
            idx1 = None
            idx2 = None
            for t_idx, tri in enumerate(triangles):
                if ia in tri and ib in tri and ic in tri:
                    idx1 = t_idx
                elif ia in tri and ib in tri and id_ in tri:
                    idx2 = t_idx
                if (idx1 is not None) and (idx2 is not None):
                    break
            if idx1 is None or idx2 is None:
                continue  # robustness
            # replace with flipped triangles (ic, id_, ia) and (id_, ic, ib)
            triangles[idx1] = [ic, id_, ia]
            triangles[idx2] = [id_, ic, ib]
            flipped += 1
    return np.asarray(triangles, dtype=int), flipped

def build_edge_map(triangles):
    """Return mapping edge-> list[triId] (each edge undirected)."""
    edge_map = defaultdict(list)
    for tidx, (i, j, k) in enumerate(triangles):
        edges = [(i, j), (j, k), (k, i)]
        for a, b in edges:
            key = tuple(sorted((a, b)))
            edge_map[key].append(tidx)
    return edge_map


def refine_mesh(vertices, triangles, points, tol_edge=0.2, tol_vertex_dist=None):
    """Refine triangular *mesh* by inserting *points* per snapping rules.

    The algorithm closely follows the description:
    1. Snap to vertex if close to three edges (two barycentric coords < tol_edge).
    2. If close to one edge (exactly one barycentric coord < tol_edge) -> split edge.
    3. Otherwise insert inside triangle and connect to its three vertices.

    Parameters
    ----------
    vertices : (N,2) ndarray
    triangles : (M,3) ndarray (int)
    points : (P,2) ndarray
    tol_edge : float
        Distance threshold (in barycentric coordinate) to consider *close* to edge.
    tol_vertex : float
        Tighter threshold for snapping to vertex.
    """
    vertices = vertices.tolist()
    triangles = triangles.tolist()
    if tol_vertex_dist is None:
        tol_vertex_dist = tol_edge

    for p in points:
        p = np.asarray(p, float)

        # find containing triangle (naive O(T))
        tri_idx = None
        bary = None
        for idx, (ia, ib, ic) in enumerate(triangles):
            a, b, c = np.asarray(vertices[ia]), np.asarray(vertices[ib]), np.asarray(vertices[ic])
            u, v, w = barycentric_coords(p, a, b, c)
            if (u >= -tol_edge) and (v >= -tol_edge) and (w >= -tol_edge):
                tri_idx = idx
                bary = np.array([u, v, w])
                break
        if tri_idx is None:
            print("Warning: point", p, "outside mesh, skipping")
            continue

        # classification
        close = bary < tol_edge  # proximity to each edge (barycentric)
        # Euclidean distance to triangle vertices for snapping decision
        ia, ib, ic = triangles[tri_idx]
        vids = [ia, ib, ic]
        tv = np.asarray([vertices[ia], vertices[ib], vertices[ic]])
        dists = np.linalg.norm(tv - p, axis=1)
        min_idx = np.argmin(dists)
        
        # DEBUG: Print distances for analysis
        #print(f"Point: {p}")
        #print(f"Vertex distances: {dists}")
        #print(f"Min distance: {dists[min_idx]} vs tol: {tol_vertex_dist}")
        
        if dists[min_idx] < tol_vertex_dist:
            # move existing vertex to the new position (snap)
            #print(f"Snapping vertex {vids[min_idx]} from {vertices[vids[min_idx]]} to {p}")
            vertices[vids[min_idx]] = p.tolist()
            continue  # no topology change
        elif np.sum(close) == 1:  # near single edge – split edge
            edge_pos = np.where(close)[0][0]
            ia, ib, ic = triangles[tri_idx]
            vids = [ia, ib, ic]
            # vertices of the edge opposite barycoord zero? Actually bary near 0 means p near edge opposite that vertex.
            # If u~0 -> edge (b,c), v~0 -> edge (c,a), w~0 -> edge (a,b)
            if edge_pos == 0:
                v_edge = (ib, ic)
                v_opposite = ia
            elif edge_pos == 1:
                v_edge = (ic, ia)
                v_opposite = ib
            else:
                v_edge = (ia, ib)
                v_opposite = ic
            # Add new vertex
            new_vid = len(vertices)
            vertices.append(p.tolist())

            # Find the second triangle sharing this edge (if any)
            edge_map = build_edge_map(triangles)
            adjacent_tris = edge_map[tuple(sorted(v_edge))]

            # Replace triangles sharing edge
            for t in adjacent_tris[::-1]:  # iterate backwards because we'll pop
                ia2, ib2, ic2 = triangles.pop(t)
                # Determine opposite vertex in this triangle
                opp = list({ia2, ib2, ic2} - set(v_edge))[0]
                # Create two new triangles
                triangles.append([v_edge[0], new_vid, opp])
                triangles.append([new_vid, v_edge[1], opp])
        else:  # interior – split triangle into 3 triangles
            ia, ib, ic = triangles.pop(tri_idx)
            new_vid = len(vertices)
            vertices.append(p.tolist())
            triangles.append([ia, ib, new_vid])
            triangles.append([ib, ic, new_vid])
            triangles.append([ic, ia, new_vid])

    return np.asarray(vertices), np.asarray(triangles, dtype=int)

# -----------------------------------------------------------
#  Demonstration
# -----------------------------------------------------------

def plotBonds(bonds, points, ax=None, color="k", ls="-", linewidth=1):
    if ax is None:
        ax = plt.gca()
    for bond in bonds:
        i,j=bond
        pi=points[i-1]
        pj=points[j-1]
        ax.plot([pi[0],pj[0]], [pi[1],pj[1]], color=color, ls=ls, linewidth=linewidth)
    

def demo_flip_edges():
    # Create manual bowtie configuration (two triangles sharing a long edge)
    vertices = np.array([
        [0,0],   # 0
        [2,0],   # 1
        [1,1],   # 2 (top center)
        [0,2],   # 3
        [2,2]    # 4
    ])
    triangles = np.array([
        [0,1,2],  # lower triangle
        [3,4,2]   # upper triangle (shares vertex 2)
    ])
    
    # Make the shared edge artificially long
    vertices[2] = [1, 0.5]  # pull center point down
    
    # Plot before flipping
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("Before edge flipping")
    ax1.triplot(vertices[:,0], vertices[:,1], triangles, 'b-')
    ax1.plot(vertices[:,0], vertices[:,1], 'ro')
    
    # Apply edge flipping
    triangles_flipped, num_flipped = flip_long_edges(vertices, triangles)
    #print(f"Flipped {num_flipped} edges")
    
    # Plot after flipping
    ax2.set_title(f"After flipping {num_flipped} edges")
    ax2.triplot(vertices[:,0], vertices[:,1], triangles_flipped, 'r-')
    ax2.plot(vertices[:,0], vertices[:,1], 'ro')
    
    plt.tight_layout()
    plt.show()

def points_along_bonds(points, bonds, n=1):
    ps = []
    for bond in bonds:
        i,j=bond
        pi=points[i-1]
        pj=points[j-1]
        d = (pj-pi)*(1./n)
        for k in range(n):
            ps.append((pi + d*(k+0.5)))
    return np.array(ps)

if __name__ == "__main__":

    points=np.array([
        [2.7734, 1.6941],
        [3.1541, 0.3693],
        [2.0854, -0.3955],
        [1.0167, 0.3694],
        [1.3975, 1.6942],
        [3.4241, 2.5581],
        [0.0000, 0.0000],
        [4.1709, -0.0000],
        [0.7771, 2.5835],
    ])

    bonds=[
            (4,7),  
            (1,2),  
            (1,6),  
            (2,8),  
            (2,3),  
            (3,4), 
            (1,5),   
            (4,5),   
            (5,9), 
    ]

    

    points[:,0] += 0.0
    points[:,1] += 2.55

    bpoints = points_along_bonds(points, bonds, n=1)
    print( "bpoints.shape ", bpoints.shape)

    points = np.vstack((points, bpoints))

    nx, ny = 4, 4
    verts0, tris0 = generate_equilateral_triangle_grid(nx, ny, L=2.0)

    verts1, tris1 = refine_mesh(verts0, tris0, points, tol_edge=0.3)
    #verts2, tris2 = refine_mesh(verts1, tris1, bpoints, tol_edge=0.3)

    fig, ax = plt.subplots(1, 1, figsize=(8,8), sharex=True, sharey=True)

    ax.triplot(verts0[:,0], verts0[:, 1], tris0,  color="gray", linewidth=0.5)
    ax.triplot(verts1[:, 0], verts1[:, 1], tris1, color="k", linewidth=0.5)
    #ax.triplot(verts2[:, 0], verts2[:, 1], tris2, color="k", linewidth=0.5)

    ax.plot(points[:, 0], points[:, 1], "ro", ms=3, label="Inserted atom pts")
    ax.plot(bpoints[:, 0], bpoints[:, 1], "go", ms=3, label="Inserted bonds pts")

    for i, point in enumerate(points):
        ax.text(point[0], point[1], str(i+1), fontsize=12, ha="center", va="center")

    plotBonds(bonds, points, ax, color="g", ls=":", linewidth=1)
    ax.set_aspect("equal")
    plt.tight_layout()


    # Apply edge flipping on refined mesh
    tris_flipped, num_flipped = flip_long_edges(verts1, tris1)
    #tris_flipped, num_flipped = flip_long_edges(verts2, tris2)
    print(f"Flipped {num_flipped} edges in refined mesh")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
    # Before flipping
    ax1.set_title("Before flipping")
    ax1.triplot(verts1[:,0], verts1[:,1], tris1, 'k-', alpha=0.5)
    ax1.plot(points[:,0], points[:,1], 'ko')
    ax1.set_aspect("equal")

    ax2.set_title(f"After flipping {num_flipped} edges")
    ax2.triplot(verts1[:,0], verts1[:,1], tris_flipped, 'k-', alpha=0.5)
    ax2.plot(points[:,0], points[:,1], 'ko')
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    plt.show()
