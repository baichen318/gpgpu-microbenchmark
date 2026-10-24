# RTX PRO 6000 Blackwell Microbenchmark

A CUDA microbenchmark suite for the **NVIDIA RTX PRO 6000 Blackwell Server
Edition** (GB202, compute capability `sm_120`). It measures the SM
configuration, the L1/L2/HBM memory hierarchy, GDDR7 bandwidth, scalar and
tensor-core throughput, the on-chip NoC, PCIe, occupancy, and sustained
power/clock behavior.

Repository: <https://github.com/baichen318/gpgpu-microbenchmark/tree/main/rtx-pro-6000-blackwell>

The accompanying report is:

- [`report/rtx-pro-6000-blackwell-architecture-report.md`](report/rtx-pro-6000-blackwell-architecture-report.md)

The report follows the question/answer/test/results/analysis format of the
original *Secrets Behind NVIDIA GPGPU* H20 study, adapted to a single-GPU
Blackwell part.

## Hardware and software used

| Component | Detail |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (GB202, `sm_120`) |
| SMs | 188 |
| Memory | 96 GB GDDR7 with ECC, 512-bit, 12,481 MHz |
| L2 | 128 MiB |
| Shared memory | 100 KiB per SM, 99 KiB opt-in per block |
| Power limit | 600 W |
| PCIe | Gen5 x16 capable; **Gen5 x8** in the test system |
| Driver / CUDA | 615.71.10 / 13.4 |
| OS | Rocky Linux 8.10 |

The code targets `sm_120`; the default build uses `sm_120a` because the FP4
block-scaled tests require it. To run on another Blackwell SKU, change the
gencode in the Makefile. The reported numbers are specific to the card above.

## Repository layout

```text
include/common.cuh          Shared timing, CSV, device-info, and PTX helpers
src/bench_device.cu         Device properties and occupancy envelope
src/bench_latency.cu        Pointer-chase L1/L2/HBM latency
src/bench_bandwidth.cu      HBM read/write/copy/triad/memset/memcpy
src/bench_cache.cu          L1/L2/shared bandwidth, bank conflicts, shared latency
src/bench_compute.cu        FP32/FP64/INT32/SFU/FP16 throughput and FMA latency
src/bench_tensor.cu         FP16/BF16/FP8/FP4 tensor-core throughput
src/bench_noc.cu            On-chip NoC latency matrix and bandwidth scaling
src/bench_pcie.cu           Pinned/pageable PCIe bandwidth
src/bench_scheduler.cu      Issue scaling and occupancy vs shared memory
src/bench_power.cu          Sustained power/clock/temperature via NVML
scripts/run_all.sh          Run the complete suite
scripts/make_figures.py     Regenerate SVG figures from the raw CSV data
scripts/monitor_gpu.sh      Standalone nvidia-smi sampler
scripts/ncu_profile.sh      Optional Nsight Compute profile
results/raw/                Raw measurements used in the report
report/                     Markdown report and SVG figures
```

## Build

```bash
make -j
```

The default build is:

```make
NVCCFLAGS ?= -O3 -std=c++17 \
  -gencode arch=compute_120a,code=sm_120a \
  -lineinfo -Xcompiler -Wall
```

The explicit `compute_120a` virtual architecture is required for the packed
FP4 block-scaled instructions (`kind::mxf4`, `kind::mxf4nvf4`, `.block_scale`,
`.scale_vec`, `.e2m1`). The shorthand `-arch=sm_120a` generated `compute_120`
PTX with this toolchain and ptxas rejected those instructions as unsupported on
`sm_120`. If you do not need FP4, you can build the other benchmarks with
`-arch=sm_120`.

## Run

```bash
./scripts/run_all.sh
```

The script runs the benchmarks sequentially, writes CSV files to
`results/raw/`, captures a system snapshot, and logs the console output. The
power test adds about 19 seconds; the full suite takes several minutes.

Individual benchmarks:

```bash
./bin/bench_device   --out results/raw/device_info.csv
./bin/bench_latency  --max-size 1G --repeat 3 --out results/raw/memory_latency.csv
./bin/bench_bandwidth --max-size 8G --repeat 5 --out results/raw/memory_bandwidth.csv
./bin/bench_cache    --repeat 5 --out results/raw/cache.csv
./bin/bench_compute  --repeat 3 --out results/raw/compute.csv
./bin/bench_tensor   --repeat 3 --out results/raw/tensor.csv
./bin/bench_noc      --repeat 3 --out-prefix results/raw/noc
./bin/bench_pcie     --repeat 10 --out results/raw/pcie.csv
./bin/bench_scheduler --repeat 3 --out results/raw/scheduler.csv
./bin/bench_power    --iters 400000000 --out results/raw/power.csv
```

## What each test measures

| Test | Question |
|---|---|
| Device | How many SMs, warps, registers, shared-memory bytes, and L2 bytes are exposed? What is the occupancy envelope? |
| Latency | What are the L1, L2, and HBM latencies and capacities? |
| Bandwidth | What is the sustained HBM bandwidth, and how does it compare with the clock-derived and marketing peaks? |
| Cache | How fast are L1, L2, and shared memory? How costly are bank conflicts? |
| Compute | What are the FP32, FP64, INT32, SFU, and FP16 throughputs and the FMA latency? |
| Tensor | What are the FP16, BF16, FP8, and FP4 tensor-core throughputs? |
| NoC | Is the on-chip network a crossbar or a distance-dependent mesh? What are its latency and bandwidth? |
| PCIe | What is the host-device bandwidth, and what link generation/width is active? |
| Scheduler | How many warps/blocks can the SM keep resident, and how does issue scale? |
| Power | Does the GPU sustain its clock, how much power does it draw, and does it throttle? |

## Selected results

| Metric | Value |
|---|---:|
| SMs / threads per SM / blocks per SM | 188 / 1,536 / 24 |
| L2 cache | 128 MiB |
| Shared memory per SM | 100 KiB |
| L1 hit latency | 22.4 ns / 53 cycles |
| L2 hit latency | 150-160 ns / 356-370 cycles |
| HBM latency | 360-369 ns / 872-891 cycles |
| Sustained HBM read | 1.54 TB/s |
| Sustained HBM write | 1.46 TB/s |
| L1 read bandwidth | 39.84 TB/s aggregate |
| L2 read bandwidth | 7.97 TB/s aggregate |
| Shared read bandwidth | 25.97 TB/s aggregate |
| FP32 FMA | 100.53 TFLOPS |
| FP64 FMA | 1.529 TFLOPS |
| FP16 / BF16 tensor | 466.6 / 466.7 TFLOPS |
| FP8 tensor | 932.7 TFLOPS |
| Packed FP4 tensor | 1.81 PFLOPS |
| NoC median round-trip | 1,254 ns |
| NoC aggregate bandwidth | 7.99 TB/s |
| PCIe Gen5 x8 | 28.1 GB/s H2D, 28.7 GB/s D2H |
| Sustained SM clock | 2,422 MHz |
| FP32 power | 330 W average, 429 W peak |
| FP4 tensor power | 245-253 W steady, 267 W peak |

See the [report](report/rtx-pro-6000-blackwell-architecture-report.md) for the
full analysis, tables, and figures.

## Regenerate figures

```bash
python3 scripts/make_figures.py
```

The script uses only the Python standard library and writes SVG files to
`report/figures/`.

## Notes and caveats

- The test system exposes the GPU at **PCIe Gen5 x8**, not x16. A x16 slot
  would roughly double the PCIe ceiling.
- ECC is enabled. ECC reduces the effective HBM payload bandwidth; the 1.54
  TB/s sustained number includes that cost.
- The GPU is a single device with no NVLink/NVSwitch. Multi-GPU tests from the
  H20 report are not applicable.
- The FP4 tensor number is a compute-only ceiling with scale factors fixed at
  1.0. A real MXFP4/NVFP4 GEMM will be lower due to scale-factor loads and
  dequantization.
- The machine is shared. No other GPU processes were visible during the
  reported runs, but clocks and bandwidth can vary with other users' activity.
- Clock locking was not used. The GPU sustained 2,422 MHz in these runs, but a
  few MHz of run-to-run variation is expected.
