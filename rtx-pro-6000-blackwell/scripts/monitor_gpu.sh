#!/usr/bin/env bash
# Sample GPU clocks, power, temperature, utilization, and PCIe link state.
# Usage: scripts/monitor_gpu.sh [interval_ms]
set -u
INTERVAL_MS="${1:-500}"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
  --format=csv -lms "${INTERVAL_MS}"
