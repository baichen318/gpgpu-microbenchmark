#!/usr/bin/bash
# Author: eecbai@ust.hk

set +x

export CUDA_VISIBLE_DEVICES=0,1
make -j`nproc`
./load
