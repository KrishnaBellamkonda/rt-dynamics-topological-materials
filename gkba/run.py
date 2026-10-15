"""
Command line driver for one GKBA quench. The quench masses and mixing are fixed
to the values used in the runs (M_pre = -3.5 to M_post = -1.0, lam = 0.4). The
interaction, grid and time stepping come from the command line. Output is a .npz.

  local     python run.py --V0 1.0 --V1 0.0  --Nk 100 --Nt 2000 --out local.npz
  nonlocal  python run.py --V0 1.0 --V1 0.25 --Nk 100 --Nt 2000 --out nonlocal.npz

Backend is set with GKBA_BACKEND=numpy or cupy.
"""
import argparse
import time
import numpy as np

import gkba

# fixed quench, mass before and after and the band mixing
M_PRE, M_POST, LAM = -3.5, -1.0, 0.4


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--V0", type=float, required=True, help="on site interaction")
    ap.add_argument("--V1", type=float, required=True,
                    help="nearest neighbour interaction, 0 is the local model")
    ap.add_argument("--Nk", type=int, default=100, help="k points per direction")
    ap.add_argument("--Nt", type=int, default=2000, help="number of time steps")
    ap.add_argument("--dt", type=float, default=0.05, help="time step")
    ap.add_argument("--nmem", type=int, default=300, help="collision memory window")
    ap.add_argument("--save_every", type=int, default=20, help="record cadence")
    ap.add_argument("--out", required=True, help="output .npz path")
    a = ap.parse_args()

    print(f"k resolved GKBA V0={a.V0} V1={a.V1} Nk={a.Nk}^2 "
          f"Nt={a.Nt} nmem={a.nmem}", flush=True)
    t0 = time.time()
    out = gkba.gkba_run_kresolved(M_PRE, M_POST, LAM, a.V0, a.V1, a.Nk, a.Nt, a.dt,
                                  n_mem=a.nmem, save_every=a.save_every, verbose=True)
    np.savez(a.out, **out)
    print(f"saved {a.out} ({time.time()-t0:.0f}s) f+_end={out['f_plus'][-1]:.4f} "
          f"pur_end={out['purity_min'][-1]:.3f} C_inst_end={out['C_inst'][-1]:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
