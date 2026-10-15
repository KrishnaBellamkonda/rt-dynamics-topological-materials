#!/bin/bash
# nonlocal model, on site plus nearest neighbour (V0 = 1.0, V1 = 0.25)
# 100 by 100 grid, 2000 steps of dt = 0.05, memory window 300
# CPU by default. for a GPU set GKBA_BACKEND=cupy
set -e
cd "$(dirname "$0")"
python ../run.py --V0 1.0 --V1 0.25 --Nk 100 --Nt 2000 --nmem 300 --save_every 20 \
    --out gkba_nonlocal.npz
