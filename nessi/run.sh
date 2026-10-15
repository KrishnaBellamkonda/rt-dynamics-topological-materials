#!/bin/bash
# run the two local KBE cases, v0 (non interacting) and v1 (interacting local)
# build first with ./build.sh. set NESSI if the shared libraries are elsewhere
set -e
cd "$(dirname "$0")"
NESSI=${NESSI:-/local/data/public/skcb2/nessi-stack}
export LD_LIBRARY_PATH=$NESSI/install/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
build/chern_kbe.x v0.inp v0
build/chern_kbe.x v1.inp v1
echo done
