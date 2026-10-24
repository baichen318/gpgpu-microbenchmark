# GPGPU Microbenchmark
The repo. collects microbenchmarks for GPGPU.

## System configuration
- NVIDIA H20 x8: `test1`, `test2`, `test4`, `test5`, `test6`.
- NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`): `rtx-pro-6000-blackwell`.

## Directory layout
- `test1`, `test2`, `test4`, `test5`, `test6`: H20 NVLink, NoC, remote-load, and
  local-load microbenchmarks from *Secrets Behind NVIDIA GPGPU*.
- `rtx-pro-6000-blackwell/`: CUDA microbenchmark suite and architecture report
  for the RTX PRO 6000 Blackwell. See
  [the report](rtx-pro-6000-blackwell/report/rtx-pro-6000-blackwell-architecture-report.md).

## Author

- Chen Bai, eecbai@ust.hk
