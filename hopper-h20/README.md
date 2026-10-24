# Hopper H20 Microbenchmarks

These are the original microbenchmarks from *Secrets Behind NVIDIA GPGPU*,
run on an NVIDIA H20 x8 system.

| Directory | Test | Description |
|---|---|---|
| `test1` | NVLink data transfer | Peer-to-peer transfer over NVLink; measures bandwidth utilization. |
| `test2` | SM-to-SM NoC | Stride sweep of SM-to-SM connectivity; infers the NoC topology. |
| `test4` | SM & NVLink bandwidth | Thread-block shmoo of peer-to-peer load/store over NVLink. |
| `test5` | Remote load | Pointer-chase ring across NVLink; remote load latency. |
| `test6` | Local load | Pointer-chase ring in local HBM; L1/L2/HBM latency. |

Each test has its own `README.md`, `Makefile`, and `exp.sh`. Test 3
(SM-to-SM network) is not present; the original report marks it as TODO.

## Related

- The Blackwell follow-up suite is in
  [`../rtx-pro-6000-blackwell/`](../rtx-pro-6000-blackwell/README.md).
- The original report is *Secrets Behind NVIDIA GPGPU*.
