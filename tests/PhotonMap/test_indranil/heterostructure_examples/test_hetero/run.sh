#!/bin/bash

# Run from this folder; cube paths in cubefiles.ini are interpreted relative to this folder
# If you want to use absolute cube paths, put them directly into cubefiles.ini

PPPATH=${PPPATH:-/home/indranil/git/ppafm}

CUBEFILES_INI=${PP_EXCITON_CUBEFILES_INI:-cubefiles.ini}
MOLECULES_INI=${PP_EXCITON_MOLECULES_INI:-molecules.ini}

if [ "$#" -ge 1 ]; then
  CUBEFILES_INI="$1"
fi

if [ "$#" -ge 2 ]; then
  MOLECULES_INI="$2"
fi

export PP_EXCITON_CUBEFILES_INI="$CUBEFILES_INI"
export PP_EXCITON_MOLECULES_INI="$MOLECULES_INI"

# Rebuild C++ libraries (so changes in cpp/GridUtils.cpp are applied)
make -C "$PPPATH/cpp" clean
make -C "$PPPATH/cpp" GU

# Dump exact Coulomb sampling points (voxel centers) used in couplings H[i,j]
# Output files are written to this folder as:
#   coulombGrid_III_JJJ_.xyz
# with points formatted as:
#   He  x y z   q    (grid of state i)
#   Ne  x y z   q    (grid of state j)
# (Older dumps may use C/O instead of He/Ne; plotting code accepts both.)
# NOTE: dumping ALL pairs can generate very large files.

# Examples:
#   PP_EXCITON_DUMP_PAIR="3,0" PP_EXCITON_PLOT=1 PP_EXCITON_PLOT_PAIR="3,0" bash run.sh
#   PP_EXCITON_DUMP_ALL=1 PP_EXCITON_PLOT=1 PP_EXCITON_PLOT_ALL=1 bash run.sh
#   PP_EXCITON_REPORT_SIJ_MULTIPOLES=1 bash run.sh

# Set to 1 to dump all unique off-diagonal pairs (for N=6 -> 15 files)
export PP_EXCITON_DUMP_ALL=${PP_EXCITON_DUMP_ALL:-0}             # 1 => dump coulombGrid_III_JJJ_.xyz for all i>j (can be huge)

# Alternatively dump only one pair, e.g. "3,0". Leave empty to disable.
export PP_EXCITON_DUMP_PAIR=${PP_EXCITON_DUMP_PAIR:-}           # "i,j" => dump only one pair (overrides the need for DUMP_ALL)

# Optional: generate a scatter plot PNG for a selected dumped pair
# Set to 1 to enable; set PP_EXCITON_PLOT_PAIR to choose which file, e.g. "3,0"
export PP_EXCITON_PLOT=${PP_EXCITON_PLOT:-0}                    # 1 => generate scatter_*.png for one or more dumped pairs
export PP_EXCITON_PLOT_PAIR=${PP_EXCITON_PLOT_PAIR:-${PP_EXCITON_DUMP_PAIR}}  # "i,j" => choose pair to plot (defaults to DUMP_PAIR)
export PP_EXCITON_PLOT_ALL=${PP_EXCITON_PLOT_ALL:-0}            # 1 => plot all dumped coulombGrid_*.xyz files (ignores PLOT_PAIR)

# Keep PP_EXCITON_DEBUG=0 unless you want verbose console prints
export PP_EXCITON_DEBUG=${PP_EXCITON_DEBUG:-0}                  # 1 => verbose exciton/coupling debug prints (can be very noisy)

export PP_EXCITON_REPORT_MULTIPOLES=${PP_EXCITON_REPORT_MULTIPOLES:-0}       # 1 => per-state dipole/quadrupole report computed from sampled transition density
export PP_EXCITON_REPORT_SIJ_MULTIPOLES=${PP_EXCITON_REPORT_SIJ_MULTIPOLES:-0}  # 1 => print per-pair estimates: Vdd and VmuQ+VQmu (meV) alongside H_ij
export PP_EXCITON_SUBSAMP=${PP_EXCITON_SUBSAMP:-6}              # subsampling factor used for Coulomb grids (speed/accuracy tradeoff)

python $PPPATH/photonMap.py \
  -c "$CUBEFILES_INI" -m "$MOLECULES_INI" \
  -R 10.0 -Z 5.0 -t s --excitons --volumetric \
  --output T_0_0

if [ "$PP_EXCITON_REPORT_MULTIPOLES" != "0" ]; then
  PYTHONPATH="$PPPATH" python3 - <<'PY'
import numpy as np

import pyProbeParticle.GridUtils as GU
import pyProbeParticle.photo as photo

subs = int(float((__import__('os').environ.get('PP_EXCITON_SUBSAMP','6') or '6')))

def unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-30:
        return v*0.0
    return v/n

def multipoles_from_density(rho, pos, lat, byCenter=True):
    rho = np.asarray(rho)
    pos = np.asarray(pos, dtype=float)
    lat = np.asarray(lat, dtype=float)
    nx, ny, nz = rho.shape
    pos0 = pos.copy()
    if byCenter:
        pos0 = pos0 + lat[0]*(nx*-0.5) + lat[1]*(ny*-0.5)

    ix, iy, iz = np.indices((nx, ny, nz), dtype=float)
    v = np.stack([ix+0.5, iy+0.5, iz+0.5], axis=-1).reshape(-1, 3)
    r = v @ lat + pos0[None, :]
    q = rho.reshape(-1)

    qsum = q.sum()
    dip  = (q[:, None] * r).sum(axis=0)

    r0 = r.mean(axis=0)
    rc = r - r0[None, :]
    dip_c = (q[:, None] * rc).sum(axis=0)

    r2 = (rc*rc).sum(axis=1)
    Q = np.zeros((3, 3))
    Q[0, 0] = (q*(3.0*rc[:, 0]*rc[:, 0] - r2)).sum()
    Q[1, 1] = (q*(3.0*rc[:, 1]*rc[:, 1] - r2)).sum()
    Q[2, 2] = (q*(3.0*rc[:, 2]*rc[:, 2] - r2)).sum()
    Q[0, 1] = (q*(3.0*rc[:, 0]*rc[:, 1])).sum(); Q[1, 0] = Q[0, 1]
    Q[0, 2] = (q*(3.0*rc[:, 0]*rc[:, 2])).sum(); Q[2, 0] = Q[0, 2]
    Q[1, 2] = (q*(3.0*rc[:, 1]*rc[:, 2])).sum(); Q[2, 1] = Q[1, 2]

    return qsum, dip, dip_c, Q

cube_paths = [l.strip() for l in open(__import__('os').environ.get('PP_EXCITON_CUBEFILES_INI','cubefiles.ini')) if l.strip() and not l.strip().startswith('#')]

rows=[]
for line in open(__import__('os').environ.get('PP_EXCITON_MOLECULES_INI','molecules.ini')):
    line=line.strip()
    if (line=='') or line.startswith('#'):
        continue
    w=line.split()
    x,y,z = float(w[0]), float(w[1]), float(w[2])
    rot_deg = float(w[3])
    Eex = float(w[6])
    icub = int(float(w[7]))
    rows.append((x,y,z,rot_deg,Eex,icub))

print('='*120)
print('MULTIPOLE REPORT (computed directly from transition density sampling points)')
print(f'  subsampling = {subs}   byCenter=True')
print('='*120)

for i,(x,y,z,rot_deg,Eex,icub) in enumerate(rows):
    cube = cube_paths[icub]
    rho,lvec,nDim,head = GU.loadCUBE(cube,trden=True)
    rho_s = photo.prepareRhoTransForCoumpling(rho, nsub=subs)

    lat = photo.makeTransformMat(rho_s.shape, np.array(lvec[1:]), angle=np.deg2rad(rot_deg))
    ex, ey, ez = unit(lat[0]), unit(lat[1]), unit(lat[2])

    qsum, dip_raw, dip_c, Q = multipoles_from_density(rho_s, (x,y,z), lat, byCenter=True)
    d  = dip_c
    du = unit(d)
    dloc = np.array([np.dot(d,ex), np.dot(d,ey), np.dot(d,ez)])
    phi = float(np.degrees(np.arctan2(dloc[1], dloc[0])))

    Qn = float(np.linalg.norm(Q))
    evals, evecs = np.linalg.eigh(Q)
    i0 = int(np.argmax(np.abs(evals)))
    qaxis = evecs[:, i0]
    qaxis = unit(qaxis)

    print(f'\nstate {i:2d}  icub={icub}  Eex={Eex:.6f} eV  pos=({x:+.3f},{y:+.3f},{z:+.3f})  rot={rot_deg:+.1f} deg')
    print(f'  axes_global:')
    print(f'    ex(long?)=({ex[0]:+.6f},{ex[1]:+.6f},{ex[2]:+.6f})')
    print(f'    ey(short?)=({ey[0]:+.6f},{ey[1]:+.6f},{ey[2]:+.6f})')
    print(f'    ez(norm )=({ez[0]:+.6f},{ez[1]:+.6f},{ez[2]:+.6f})')
    print(f'  monopole Q = {qsum:+.6e} (should be ~0 for a perfect transition density)')
    print(f'  dipole_centered (global): ({d[0]:+.6e},{d[1]:+.6e},{d[2]:+.6e}) |d|={np.linalg.norm(d):.6e} dir=({du[0]:+.6f},{du[1]:+.6f},{du[2]:+.6f})')
    print(f'  dipole_centered (local):  (mu_x={dloc[0]:+.6e}, mu_y={dloc[1]:+.6e}, mu_z={dloc[2]:+.6e})  in-plane angle phi=atan2(mu_y,mu_x)={phi:+.2f} deg')
    print(f'  quadrupole (centered, cartesian traceless) |Q|={Qn:.6e}  principal_axis=({qaxis[0]:+.6f},{qaxis[1]:+.6f},{qaxis[2]:+.6f})  eig={evals[i0]:+.6e}')

print('\nDone.')
PY
fi

if [ "$PP_EXCITON_PLOT" != "0" ]; then
  if [ "$PP_EXCITON_PLOT_ALL" != "0" ]; then
    python3 - <<'PY'
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_xyz(fname):
    sym=[]; x=[]; y=[]; z=[]; q=[]
    with open(fname) as f:
        n = int(f.readline().strip())
        f.readline()  # comment
        f.readline()  # U 0 0 0
        for line in f:
            s, xs, ys, zs, qs = line.split()
            sym.append(s)
            x.append(float(xs)); y.append(float(ys)); z.append(float(zs)); q.append(float(qs))
    sym=np.array(sym)
    x=np.array(x); y=np.array(y); z=np.array(z); q=np.array(q)
    return sym, x, y, z, q

def dipole_from_points(x,y,z,q,mask):
    r = np.stack([x[mask], y[mask], z[mask]], axis=1)
    qq = q[mask]
    dip = (qq[:,None] * r).sum(axis=0)
    center = r.mean(axis=0)
    return dip, center

fnames = sorted([f for f in os.listdir('.') if f.startswith('coulombGrid_') and f.endswith('_.xyz')])
if len(fnames) == 0:
    raise SystemExit('No coulombGrid_*.xyz files found to plot')

for fname in fnames:
    sym, x, y, z, q = load_xyz(fname)
    mC = (sym=='C') | (sym=='He')
    mO = (sym=='O') | (sym=='Ne')
    dipC, cenC = dipole_from_points(x,y,z,q,mC)
    dipO, cenO = dipole_from_points(x,y,z,q,mO)

    plt.figure(figsize=(7,7))
    plt.scatter(x[mC], y[mC], s=0.15, c=np.sign(q[mC]), cmap='bwr', alpha=0.25)
    plt.scatter(x[mO], y[mO], s=0.15, c=np.sign(q[mO]), cmap='bwr', alpha=0.25)

    # Overlay dipole arrows (no scaling of coordinates; arrow length is dipole component in same units)
    plt.arrow(cenC[0], cenC[1], dipC[0], dipC[1], color='k', width=0.0, head_width=0.3, length_includes_head=True)
    plt.arrow(cenO[0], cenO[1], dipO[0], dipO[1], color='k', width=0.0, head_width=0.3, length_includes_head=True)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x (A)'); plt.ylabel('y (A)')
    plt.title(fname + ' (xy)')
    out = fname.replace('coulombGrid_', 'scatter_').replace('_.xyz','_xy.png')
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print('[PLOT] saved', out, ' dipC=', dipC, ' dipO=', dipO)
PY
  elif [ "$PP_EXCITON_PLOT_PAIR" != "" ]; then
    python3 - <<'PY'
import os
import numpy as np

pair = os.environ.get('PP_EXCITON_PLOT_PAIR','').strip().replace(';',',').replace(' ', '')
pp   = pair.split(',')
if (len(pp) < 2) or (not pp[0].lstrip('-').isdigit()) or (not pp[1].lstrip('-').isdigit()):
    raise SystemExit('PP_EXCITON_PLOT_PAIR must be like "3,0"')
i = int(pp[0]); j = int(pp[1])
if j > i:
    i, j = j, i

fname = f"coulombGrid_{i:03d}_{j:03d}_.xyz"
if not os.path.isfile(fname):
    raise SystemExit(f"Missing dump file: {fname} (did you enable PP_EXCITON_DUMP_PAIR or PP_EXCITON_DUMP_ALL?)")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sym=[]; x=[]; y=[]; z=[]; q=[]
with open(fname) as f:
    n = int(f.readline().strip())
    f.readline()  # comment
    f.readline()  # U 0 0 0
    for line in f:
        s, xs, ys, zs, qs = line.split()
        sym.append(s)
        x.append(float(xs)); y.append(float(ys)); z.append(float(zs)); q.append(float(qs))

sym=np.array(sym)
x=np.array(x); y=np.array(y); z=np.array(z); q=np.array(q)

mC = (sym=='C') | (sym=='He')
mO = (sym=='O') | (sym=='Ne')

rC = np.stack([x[mC], y[mC], z[mC]], axis=1)
rO = np.stack([x[mO], y[mO], z[mO]], axis=1)

dipC = (q[mC,None] * rC).sum(axis=0)
dipO = (q[mO,None] * rO).sum(axis=0)

cenC = rC.mean(axis=0)
cenO = rO.mean(axis=0)

plt.figure(figsize=(7,7))
plt.scatter(x[mC], y[mC], s=0.15, c=np.sign(q[mC]), cmap='bwr', alpha=0.25)
plt.scatter(x[mO], y[mO], s=0.15, c=np.sign(q[mO]), cmap='bwr', alpha=0.25)

# Overlay dipole arrows (no scaling)
plt.arrow(cenC[0], cenC[1], dipC[0], dipC[1], color='k', width=0.0, head_width=0.3, length_includes_head=True)
plt.arrow(cenO[0], cenO[1], dipO[0], dipO[1], color='k', width=0.0, head_width=0.3, length_includes_head=True)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x (A)'); plt.ylabel('y (A)')
plt.title(fname + ' (xy)')
out = f"scatter_{i:03d}_{j:03d}_xy.png"
plt.tight_layout()
plt.savefig(out, dpi=200)
print('[PLOT] saved', out)
PY
  fi
fi


: <<'EXAMPLES'
Examples:

  bash run.sh

  bash run.sh cubefiles.ini molecules.ini

  PP_EXCITON_DUMP_ALL=1 bash run.sh

  PP_EXCITON_DUMP_PAIR=3,0 PP_EXCITON_PLOT=1 bash run.sh

  PP_EXCITON_REPORT_MULTIPOLES=1 PP_EXCITON_SUBSAMP=6 bash run.sh

  PP_EXCITON_DEBUG=1 PP_EXCITON_DUMP_PAIR=3,0 PP_EXCITON_PLOT=1 PP_EXCITON_REPORT_MULTIPOLES=1 bash run.sh

  PP_EXCITON_DEBUG=1 PP_EXCITON_REPORT_MULTIPOLES=1 PP_EXCITON_DUMP_PAIR=3,0 PP_EXCITON_PLOT=1 PP_EXCITON_SUBSAMP=6 bash run.sh

 PP_EXCITON_DUMP_ALL=1 PP_EXCITON_PLOT=1 PP_EXCITON_PLOT_ALL=1 PP_EXCITON_DEBUG=1 PP_EXCITON_REPORT_MULTIPOLES=1 PP_EXCITON_REPORT_SIJ_MULTIPOLES=1 bash run.sh
python3 /home/indranil/git/ppafm/photonMap.py \
  -w /home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/ \
  -m molecules.ini -c cubefiles.ini --excitons --volumetric \
  --siteshift-cubes siteshift_cubes.ini --site-shifts electrostatics.ini \
  --siteshift-interp cubic --siteshift-chunk 100000 --siteshift-json site_shifts.json \
  -R 10.0 -Z 6.0 -t s --output out_with_site_shifts

EXAMPLES
