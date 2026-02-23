#!/usr/bin/python -u

import os
import sys
import glob
import argparse
import math
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def _read_bracket_block(lines, i0):
    n = len(lines)
    i = i0
    while i < n and ('[' not in lines[i]):
        i += 1
    if i >= n:
        return None, n

    buf = []
    bal = 0
    started = False
    while i < n:
        s = lines[i].strip()
        if s:
            buf.append(s)
            bal += s.count('[') - s.count(']')
            started = True
            if started and bal == 0:
                return ' '.join(buf), i + 1
        i += 1

    return None, n


def _parse_numpy_vector(block):
    s = block.strip()
    if not (s.startswith('[') and s.endswith(']')):
        print("ERROR: vector block does not start/end with brackets")
        sys.exit(2)
    s = s.replace('[', ' ').replace(']', ' ')
    v = np.fromstring(s, sep=' ', dtype=float)
    return v


def _parse_numpy_matrix(block):
    s = block.strip()
    if not (s.startswith('[') and s.endswith(']')):
        print("ERROR: matrix block does not start/end with brackets")
        sys.exit(2)
    s = s.replace('\n', ' ')
    row_txts = []
    buf = []
    bal = 0
    for ch in s:
        if ch == '[':
            bal += 1
        elif ch == ']':
            bal -= 1

        buf.append(ch)

        if bal == 1 and ch == ']':
            row_txts.append(''.join(buf))
            buf = []

    rows = []
    for rt in row_txts:
        rt = rt.replace('[', ' ').replace(']', ' ')
        r = np.fromstring(rt, sep=' ', dtype=float)
        if r.size > 0:
            rows.append(r)

    if len(rows) == 0:
        print("ERROR: failed to parse any rows from matrix block")
        sys.exit(2)

    ncol = int(max(r.size for r in rows))
    for i, r in enumerate(rows):
        if r.size != ncol:
            print(f"ERROR: non-rectangular matrix parsed, row {i} has {r.size} cols, expected {ncol}")
            sys.exit(2)

    M = np.vstack(rows)
    return M


def load_ham_eigs(ham_path):
    with open(ham_path, 'r') as f:
        lines = f.readlines()

    b0, i = _read_bracket_block(lines, 0)
    b1, i = _read_bracket_block(lines, i)
    b2, i = _read_bracket_block(lines, i)

    if (b0 is None) or (b1 is None) or (b2 is None):
        print(f"ERROR: failed to parse '{ham_path}'")
        sys.exit(2)

    H = _parse_numpy_matrix(b0)
    eigEs = _parse_numpy_vector(b1)
    eigVs = _parse_numpy_matrix(b2)

    if H.shape[0] != H.shape[1]:
        print(f"ERROR: H is not square in '{ham_path}', shape={H.shape}")
        sys.exit(2)

    if eigEs.shape[0] != H.shape[0]:
        print(f"ERROR: eigEs size mismatch in '{ham_path}', len(eigEs)={eigEs.shape[0]} vs n={H.shape[0]}")
        sys.exit(2)

    if eigVs.shape != H.shape:
        print(f"ERROR: eigVs shape mismatch in '{ham_path}', eigVs={eigVs.shape} vs H={H.shape}")
        sys.exit(2)

    return H, eigEs, eigVs


def find_ham_file(folder, ham_name):
    p0 = os.path.join(folder, ham_name)
    if os.path.isfile(p0):
        return p0

    cands = sorted(glob.glob(os.path.join(folder, '*_0.ham')))
    if len(cands) > 0:
        return cands[0]

    return None


def infer_nmol_from_H(H, eps):
    n = H.shape[0]
    if H.shape[0] != H.shape[1]:
        print(f"ERROR: cannot infer n_mol from non-square H, shape={H.shape}")
        sys.exit(2)

    # We assume the basis is ordered as contiguous blocks per molecule.
    # In this exciton construction there are typically no *intra-molecule* couplings,
    # so each diagonal block should be (approximately) diagonal.
    # We find the largest block size b (dividing n) for which every diagonal block
    # has off-diagonal magnitude <= eps, then infer n_mol = n/b.
    divisors = [b for b in range(1, n + 1) if (n % b) == 0]
    best_b = None
    for b in divisors:
        ok = True
        for i0 in range(0, n, b):
            B = H[i0:i0 + b, i0:i0 + b]
            off = B - np.diag(np.diag(B))
            if float(np.max(np.abs(off))) > eps:
                ok = False
                break
        if ok:
            best_b = b

    if best_b is None:
        print(f"ERROR: cannot infer n_mol (no diagonal-only block size found, eps={eps:g})")
        sys.exit(2)

    n_mol = int(n // best_b)
    if n_mol < 1:
        print(f"ERROR: inferred invalid n_mol={n_mol}")
        sys.exit(2)

    return n_mol


def plot_splitting(
    H,
    eigEs,
    eigVs,
    out_path,
    title,
    n_mol,
    wcut,
    topk,
    figsize,
    dpi,
    annotate,
    annotate_fmt,
    color_mode,
    connector_bicolor,
    cmap_name,
    line_halfwidth,
    conn_lw_min,
    conn_lw_max,
    conn_lw_gamma,
    infer_eps,
):
    n = H.shape[0]

    if n_mol == 0:
        n_mol = infer_nmol_from_H(H, infer_eps)

    if n_mol < 1:
        print("ERROR: n_mol must be >=1 (or use --n-mol 0 to infer)")
        sys.exit(2)

    if (n % n_mol) != 0:
        print(f"ERROR: total basis size n={n} not divisible by n_mol={n_mol}")
        sys.exit(2)

    n_per = n // n_mol

    if n_mol == 2:
        xs = np.array([0.0, 1.0])
        x_c = 0.5
    else:
        # Put molecules in vertical stacks (one x per molecule) and insert a gap
        # column in the middle for the coupled eigen-energies.
        mid = int(n_mol // 2)
        xs = np.arange(n_mol, dtype=float)
        xs[mid:] += 1.0
        x_c = float(mid)
    w = float(line_halfwidth)

    mol_cmap = plt.get_cmap('tab10')
    mol_colors = [mol_cmap(i % 10) for i in range(n_mol)]
    basis_cmap = plt.get_cmap(cmap_name, n)
    eig_cmap = plt.get_cmap(cmap_name, n)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    Ediag = np.diag(H).copy()

    ymin = min(float(np.min(Ediag)), float(np.min(eigEs)))
    ymax = max(float(np.max(Ediag)), float(np.max(eigEs)))
    yspan = (ymax - ymin + 1e-9)
    yoff = 0.012 * yspan

    if annotate_fmt is None:
        annotate_fmt = '{:.4f}'

    def _pick_basis_color(j, im):
        if (color_mode == 'basis') or (color_mode == 'both'):
            return basis_cmap(j)
        if color_mode == 'molecule':
            return mol_colors[im]
        return (0.25, 0.25, 0.25, 1.0)

    def _pick_eig_color(ie):
        if (color_mode == 'eigen') or (color_mode == 'both'):
            return eig_cmap(ie)
        return (0.0, 0.0, 0.0, 1.0)

    for im in range(n_mol):
        x0 = xs[im]
        Es = Ediag[im * n_per:(im + 1) * n_per]
        for k, e in enumerate(Es):
            j = im * n_per + k
            col = _pick_basis_color(j, im)
            ax.hlines(e, x0 - w, x0 + w, color=col, lw=2.0)
            if annotate:
                if x0 <= x_c:
                    xt, ha = x0 - w - 0.05, 'right'
                else:
                    xt, ha = x0 + w + 0.05, 'left'
                ax.text(xt, float(e), annotate_fmt.format(float(e)), ha=ha, va='center', fontsize=8, color=col)

    for ie, e in enumerate(eigEs):
        col = _pick_eig_color(ie)
        ax.hlines(e, x_c - w, x_c + w, color=col, lw=2.2)
        if annotate:
            dy = yoff * (1.0 + 0.35 * (ie % 3))
            ax.text(x_c, float(e) + dy, annotate_fmt.format(float(e)), ha='center', va='bottom', fontsize=8, color=col)

    for ie in range(n):
        v = eigVs[ie].copy()
        wts = (np.abs(v) ** 2)
        if topk > 0:
            js = np.argsort(-wts)[:topk]
        else:
            js = np.arange(n)

        for j in js:
            wt = float(wts[j])
            if wt < wcut:
                continue

            im = int(j // n_per)
            x0 = xs[im]
            e0 = float(Ediag[j])
            e1 = float(eigEs[ie])

            if x0 < x_c:
                xa, xb = x0 + w, x_c - w
            else:
                xa, xb = x0 - w, x_c + w

            if conn_lw_max <= conn_lw_min:
                print(f"ERROR: conn_lw_max must be > conn_lw_min, got {conn_lw_max} <= {conn_lw_min}")
                sys.exit(2)
            if conn_lw_gamma <= 0.0:
                print(f"ERROR: conn_lw_gamma must be > 0, got {conn_lw_gamma}")
                sys.exit(2)

            lw = float(conn_lw_min + (conn_lw_max - conn_lw_min) * (wt ** conn_lw_gamma))
            col_basis = _pick_basis_color(j, im)
            col_eig = _pick_eig_color(ie)
            if connector_bicolor:
                xm = 0.5 * (xa + xb)
                ym = 0.5 * (e0 + e1)
                ax.plot([xa, xm], [e0, ym], linestyle=':', color=col_basis, lw=lw, alpha=0.80)
                ax.plot([xm, xb], [ym, e1], linestyle=':', color=col_eig, lw=lw, alpha=0.80)
            else:
                if color_mode == 'basis':
                    col = col_basis
                elif color_mode == 'eigen':
                    col = col_eig
                else:
                    col = mol_colors[im]
                ax.plot([xa, xb], [e0, e1], linestyle=':', color=col, lw=lw, alpha=0.75)

    if n_mol == 2:
        ax.set_xlim(-0.7, 1.7)
    else:
        ax.set_xlim(float(np.min(xs)) - 0.7, float(max(np.max(xs), x_c)) + 0.7)

    pad = 0.05 * yspan
    ax.set_ylim(ymin - pad, ymax + pad)

    if n_mol == 2:
        ax.set_xticks([xs[0], x_c, xs[-1]])
        ax.set_xticklabels(['mol 1', 'coupled', 'mol 2'])
    else:
        order = np.argsort(xs)
        xt = [float(xs[i]) for i in order] + [float(x_c)]
        xl = [f'mol {int(i) + 1}' for i in order] + ['coupled']
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, rotation=0)

    ax.set_ylabel('Energy (eV)')
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='dirs', nargs='*', default=[], help='One or more output folders (e.g. T_-1_-1)')
    parser.add_argument('--root', default='.', help='Root directory containing T_*_* folders')
    parser.add_argument('--glob', dest='globpat', default='', help='Glob pattern under --root (e.g. "T_*_*")')
    parser.add_argument('--ham-name', default='molecules.ini_0.ham')
    parser.add_argument('--out', default='energy_splitting.png')
    parser.add_argument('--n-mol', type=int, default=2)
    parser.add_argument('--wcut', type=float, default=0.10)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--figsize', type=float, nargs=2, default=(6.5, 5.0))
    parser.add_argument('--annotate', action='store_true', default=False)
    parser.add_argument('--annotate-fmt', default='{:.4f}')
    parser.add_argument('--color-mode', choices=['molecule', 'basis', 'eigen', 'both'], default='molecule')
    parser.add_argument('--connector-bicolor', action='store_true', default=False)
    parser.add_argument('--cmap', default='tab10')
    parser.add_argument('--line-halfwidth', type=float, default=0.22)
    parser.add_argument('--conn-lw-min', type=float, default=0.15)
    parser.add_argument('--conn-lw-max', type=float, default=1.30)
    parser.add_argument('--conn-lw-gamma', type=float, default=0.30)
    parser.add_argument('--infer-eps', type=float, default=1e-6)

    args = parser.parse_args(argv)

    folders = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _resolve_root(root):
        if os.path.isabs(root):
            return root
        cand_cwd = os.path.abspath(os.path.join(os.getcwd(), root))
        if os.path.isdir(cand_cwd):
            return cand_cwd
        cand_script = os.path.abspath(os.path.join(script_dir, root))
        if os.path.isdir(cand_script):
            return cand_script
        return cand_cwd

    if len(args.dirs) > 0:
        folders = [os.path.abspath(d) for d in args.dirs]
    elif args.globpat:
        root_abs = _resolve_root(args.root)
        folders = sorted(glob.glob(os.path.join(root_abs, args.globpat)))
        folders = [p for p in folders if os.path.isdir(p)]
    else:
        print("ERROR: provide --dir ... or --glob ...")
        return 2

    if len(folders) == 0:
        print("ERROR: no folders found")
        return 2

    for folder in folders:
        if not os.path.isdir(folder):
            print(f"ERROR: not a directory: {folder}")
            return 2

        ham_path = find_ham_file(folder, args.ham_name)
        if ham_path is None:
            print(f"ERROR: cannot find ham file in {folder}")
            return 2

        H, eigEs, eigVs = load_ham_eigs(ham_path)

        out_path = os.path.join(folder, args.out)
        title = os.path.basename(folder)
        plot_splitting(
            H,
            eigEs,
            eigVs,
            out_path=out_path,
            title=title,
            n_mol=args.n_mol,
            wcut=args.wcut,
            topk=args.topk,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            annotate=args.annotate,
            annotate_fmt=args.annotate_fmt,
            color_mode=args.color_mode,
            connector_bicolor=args.connector_bicolor,
            cmap_name=args.cmap,
            line_halfwidth=args.line_halfwidth,
            conn_lw_min=args.conn_lw_min,
            conn_lw_max=args.conn_lw_max,
            conn_lw_gamma=args.conn_lw_gamma,
            infer_eps=args.infer_eps,
        )
        print(f"saved: {out_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())



'''
python3 plot_exciton_splitting.py \
  --dir tests/PhotonMap/test_indranil/heterostructure_examples/test_hetero/T_-1_-1 \
  --out energy_splitting.png \
  --n-mol 2 \
  --wcut 0.10 \
  --topk 2


python3 plot_exciton_splitting.py \
  --root tests/PhotonMap/test_indranil/heterostructure_examples/test_hetero \
  --glob "T_*_*" \
  --out energy_splitting.png \
  --n-mol 2



python3 plot_exciton_splitting.py \
  --dir T_-1_-1 \
  --out energy_splitting.png \
  --n-mol 2 \
  --annotate \
  --color-mode both \
  --connector-bicolor \
  --topk 2 \
  --wcut 0.10


python3 plot_exciton_splitting.py \
  --dir T_-1_-1 \
  --out energy_splitting.png \
  --n-mol 0 \
  --annotate \
  --color-mode both \
  --connector-bicolor \
  --topk 0 \
  --wcut 0.001 \
  --conn-lw-min 0.10 \
  --conn-lw-max 1.20 \
  --conn-lw-gamma 0.30
'''