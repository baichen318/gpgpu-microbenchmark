#!/usr/bin/env bash
# Run the complete RTX PRO 6000 Blackwell microbenchmark suite.
set -u

cd "$(dirname "$0")/.."
export PATH="/usr/local/cuda/bin:${PATH}"
mkdir -p results/raw report/figures bin

RUN_LOG="results/raw/run.log"
: > "${RUN_LOG}"

log() {
  echo "$@" | tee -a "${RUN_LOG}"
}

run() {
  log ""
  log "== $* =="
  "$@" 2>&1 | tee -a "${RUN_LOG}"
  local rc=${PIPESTATUS[0]}
  if [ "${rc}" -ne 0 ]; then
    log "WARNING: command failed with exit code ${rc}: $*"
  fi
  return 0
}

log "RTX PRO 6000 Blackwell microbenchmark"
log "date: $(date -Is)"
{
  echo "=== uname ==="
  uname -a
  echo "=== nvidia-smi -q ==="
  nvidia-smi -q
  echo "=== nvidia-smi topo ==="
  nvidia-smi topo -m
  echo "=== lscpu ==="
  lscpu
  echo "=== free ==="
  free -h
} > results/raw/system_info.txt 2>&1

run ./bin/bench_device --out results/raw/device_info.csv

# Memory hierarchy latency.  The full ring walk is required for a large
# working set; a short walk would underestimate the cache capacity.
run ./bin/bench_latency --max-size 1G --repeat 3 \
  --out results/raw/memory_latency.csv
run ./bin/bench_latency --max-size 1G --repeat 2 --stride-mode prime \
  --out results/raw/memory_latency_prime.csv

run ./bin/bench_bandwidth --max-size 8G --repeat 5 \
  --out results/raw/memory_bandwidth.csv
run ./bin/bench_cache --repeat 5 --out results/raw/cache.csv
run ./bin/bench_compute --repeat 3 --out results/raw/compute.csv

run ./bin/bench_tensor --repeat 3 --out results/raw/tensor.csv

# Steady-state tensor-core power with a long FP4 kernel and a monitor.
(
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -lms 500 > results/raw/tensor_power_monitor.csv 2>&1
) &
MON_PID=$!
run ./bin/bench_tensor --power
kill "${MON_PID}" 2>/dev/null || true
wait "${MON_PID}" 2>/dev/null || true

# PCIe test with a link-state monitor in the background.
(
  nvidia-smi --query-gpu=timestamp,pcie.link.gen.current,pcie.link.width.current \
    --format=csv -lms 100 > results/raw/pcie_link_monitor.csv 2>&1
) &
MON_PID=$!
run ./bin/bench_pcie --repeat 10 --out results/raw/pcie.csv
kill "${MON_PID}" 2>/dev/null || true
wait "${MON_PID}" 2>/dev/null || true

run ./bin/bench_noc --repeat 3 --out-prefix results/raw/noc
run ./bin/bench_scheduler --repeat 3 --out results/raw/scheduler.csv
run ./bin/bench_power --iters 400000000 --out results/raw/power.csv

nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.memory,power.limit,pcie.link.gen.max,pcie.link.width.max \
  --format=csv > results/raw/system_summary.csv

log ""
log "All benchmarks finished.  Raw data is in results/raw/."
