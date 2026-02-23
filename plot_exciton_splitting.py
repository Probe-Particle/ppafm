#!/usr/bin/python -u

import os
import sys
import glob
import argparse
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


def plot_splitting(H, eigEs, eigVs, out_path, title, n_mol, wcut, topk, figsize, dpi):
    n = H.shape[0]

    if n_mol < 1:
        print("ERROR: n_mol must be >=1")
        sys.exit(2)

    if (n % n_mol) != 0:
        print(f"ERROR: total basis size n={n} not divisible by n_mol={n_mol}")
        sys.exit(2)

    n_per = n // n_mol

    xs = np.arange(n_mol, dtype=float)
    x_c = 0.5 * (xs[0] + xs[-1])
    w = 0.22

    cmap = plt.get_cmap('tab10')
    mol_colors = [cmap(i % 10) for i in range(n_mol)]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    Ediag = np.diag(H).copy()

    for im in range(n_mol):
        x0 = xs[im]
        Es = Ediag[im * n_per:(im + 1) * n_per]
        for k, e in enumerate(Es):
            ax.hlines(e, x0 - w, x0 + w, color=mol_colors[im], lw=2.0)

    for e in eigEs:
        ax.hlines(e, x_c - w, x_c + w, color='k', lw=2.2)

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

            lw = 0.5 + 3.0 * wt
            ax.plot([xa, xb], [e0, e1], linestyle=':', color=mol_colors[im], lw=lw, alpha=0.75)

    ax.set_xlim(float(xs[0]) - 0.7, float(xs[-1]) + 0.7)
    ymin = min(float(np.min(Ediag)), float(np.min(eigEs)))
    ymax = max(float(np.max(Ediag)), float(np.max(eigEs)))
    pad = 0.05 * (ymax - ymin + 1e-9)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xticks([xs[0], x_c, xs[-1]] if n_mol > 1 else [x_c])
    if n_mol > 1:
        ax.set_xticklabels(['mol 1', 'coupled', f'mol {n_mol}'])
    else:
        ax.set_xticklabels(['coupled'])

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

    args = parser.parse_args(argv)

    folders = []
    if len(args.dirs) > 0:
        folders = [os.path.abspath(d) for d in args.dirs]
    elif args.globpat:
        folders = sorted(glob.glob(os.path.join(os.path.abspath(args.root), args.globpat)))
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

'''