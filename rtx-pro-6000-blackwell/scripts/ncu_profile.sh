#!/usr/bin/env bash
# Optional Nsight Compute profile.  ncu often needs elevated performance-counter
# permissions; if it fails with ERR_NVGPUCTRPERM, run it as root or adjust
# NVreg_RestrictProfilingToAdminUsers.
set -eu
cd "$(dirname "$0")/.."
mkdir -p results/ncu
export PATH="/usr/local/cuda/bin:${PATH}"

METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active"

ncu --target-processes all --metrics "${METRICS}" \
  -o results/ncu/cache ./bin/bench_cache --repeat 1 || true
ncu --target-processes all --metrics "${METRICS}" \
  -o results/ncu/noc ./bin/bench_noc --repeat 1 --out-prefix results/ncu/noc || true
