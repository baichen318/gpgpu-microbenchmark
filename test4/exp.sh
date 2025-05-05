#!/usr/bin/bash
# Author: eecbai@ust.hk

set +x

export CUDA_VISIBLE_DEVICES=2,3
make -j`nproc`
./p2p-nvlink
