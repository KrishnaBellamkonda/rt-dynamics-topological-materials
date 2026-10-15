#!/bin/bash
# build the NESSi KBE driver. point NESSI at your NESSi install
#   NESSI=/path/to/nessi-stack ./build.sh
set -e
cd "$(dirname "$0")"
NESSI=${NESSI:-/local/data/public/skcb2/nessi-stack}
PROG=$NESSI/nessi/examples/programs
mkdir -p build
mpicxx -std=c++11 -O3 -fopenmp \
  -I$NESSI/install/include -I/usr/include/eigen3 -I$PROG \
  chern_kbe.cpp -o build/chern_kbe.x \
  -L$NESSI/install/lib -Wl,-rpath,$NESSI/install/lib \
  -lcntr -lgomp -lpthread
echo "built build/chern_kbe.x"
