// HBM and cache bandwidth microbenchmarks.
#include "common.cuh"

#include <cinttypes>

namespace {

__global__ void ReadKernel(const float4* __restrict__ src, float4* __restrict__ sink,
                           size_t n4, int iters) {
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int it = 0; it < iters; ++it) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
      const float4 v = bench::LoadGlobalFloat4(&src[i]);
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
  }
  if (acc.x == -1.0f) *sink = acc;
}

__global__ void WriteKernel(float4* __restrict__ dst, size_t n4, int iters,
                            float4 value) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int it = 0; it < iters; ++it) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
      bench::StoreGlobalFloat4(&dst[i], value);
    }
  }
}

__global__ void CopyKernel(const float4* __restrict__ src, float4* __restrict__ dst,
                           size_t n4, int iters) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int it = 0; it < iters; ++it) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
      bench::StoreGlobalFloat4(&dst[i], bench::LoadGlobalFloat4(&src[i]));
    }
  }
}

__global__ void TriadKernel(const float4* __restrict__ a, const float4* __restrict__ b,
                            float4* __restrict__ dst, size_t n4, int iters) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int it = 0; it < iters; ++it) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
      const float4 x = bench::LoadGlobalFloat4(&a[i]);
      const float4 y = bench::LoadGlobalFloat4(&b[i]);
      bench::StoreGlobalFloat4(
          &dst[i], make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w));
    }
  }
}

int ClampBlocks(size_t n4, int threads, int sms) {
  size_t blocks = (n4 + threads - 1) / threads;
  const size_t cap = (size_t)sms * 8;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;
  return (int)blocks;
}

int WarmIters(size_t bytes) {
  const double est_ms = (double)bytes / (1.5e12) * 1.0e3;
  int iters = (int)(50.0 / std::max(est_ms, 1.0e-6));
  if (iters < 10) iters = 10;
  if (iters > 20000) iters = 20000;
  return iters;
}

enum class Op { kRead, kWrite, kCopy, kTriad, kMemcpyD2D, kMemset };

const char* OpName(Op op) {
  switch (op) {
    case Op::kRead: return "read";
    case Op::kWrite: return "write";
    case Op::kCopy: return "copy";
    case Op::kTriad: return "triad";
    case Op::kMemcpyD2D: return "memcpy_d2d";
    case Op::kMemset: return "memset";
  }
  return "unknown";
}

size_t BytesMoved(Op op, size_t bytes) {
  switch (op) {
    case Op::kRead: return bytes;
    case Op::kWrite: return bytes;
    case Op::kCopy: return 2 * bytes;
    case Op::kTriad: return 3 * bytes;
    case Op::kMemcpyD2D: return 2 * bytes;
    case Op::kMemset: return bytes;
  }
  return bytes;
}

}  // namespace

int main(int argc, char** argv) {
  const int device = bench::GetIntArg(argc, argv, "--device", 0);
  const size_t max_size = bench::GetSizeArg(argc, argv, "--max-size", 8ull << 30);
  const int repeat = bench::GetIntArg(argc, argv, "--repeat", 5);
  const std::string out = bench::GetArg(argc, argv, "--out", "");

  CHECK_CUDA(cudaSetDevice(device));
  const bench::DeviceInfo info = bench::GetDeviceInfo(device);
  const double theoretical = bench::TheoreticalHbmBandwidthGbps(info);
  std::printf("GPU: %s, SMs=%d, L2=%d bytes, theoretical HBM=%.1f GB/s\n",
              info.prop.name, info.prop.multiProcessorCount, info.prop.l2CacheSize,
              theoretical);

  float4* d_a = nullptr;
  float4* d_b = nullptr;
  float4* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, max_size));
  CHECK_CUDA(cudaMalloc(&d_b, max_size));
  CHECK_CUDA(cudaMalloc(&d_c, max_size));
  CHECK_CUDA(cudaMemset(d_a, 1, max_size));
  CHECK_CUDA(cudaMemset(d_b, 2, max_size));

  const size_t flush_bytes = std::max<size_t>(2ull * info.prop.l2CacheSize,
                                               256ull << 20);
  void* flush_buf = nullptr;
  CHECK_CUDA(cudaMalloc(&flush_buf, flush_bytes));

  std::ofstream csv;
  if (!out.empty()) {
    csv.open(out);
    if (!csv) {
      std::fprintf(stderr, "Cannot open %s\n", out.c_str());
      return EXIT_FAILURE;
    }
    csv << "operation,state,size_bytes,iterations,time_ms,bandwidth_gbps,"
           "percent_of_theoretical\n";
  }

  const size_t sizes[] = {
      4ull << 20,   16ull << 20,  64ull << 20,  128ull << 20,
      256ull << 20, 512ull << 20, 1ull << 30,   2ull << 30,
      4ull << 30,   8ull << 30,
  };
  const Op ops[] = {Op::kRead, Op::kWrite, Op::kCopy, Op::kTriad,
                    Op::kMemcpyD2D, Op::kMemset};

  std::printf("%-12s %-6s %12s %10s %12s %12s %8s\n", "op", "state", "size",
              "iters", "time_ms", "GB/s", "%peak");

  for (size_t size : sizes) {
    if (size > max_size) continue;
    const size_t n4 = size / sizeof(float4);
    if (n4 == 0) continue;
    const int threads = 256;
    const int blocks = ClampBlocks(n4, threads, info.prop.multiProcessorCount);

    for (Op op : ops) {
      for (int cold = 0; cold < 2; ++cold) {
        if (op == Op::kMemcpyD2D && cold) continue;  // no cache distinction
        if (op == Op::kMemset && cold) continue;
        const int iters = cold ? 1 : WarmIters(size);
        std::vector<double> times;
        for (int r = 0; r < (cold ? std::max(repeat, 7) : repeat); ++r) {
          if (cold) {
            bench::FlushL2(flush_buf, flush_bytes);
            CHECK_CUDA(cudaDeviceSynchronize());
          }
          bench::GpuTimer timer;
          timer.Start();
          switch (op) {
            case Op::kRead:
              ReadKernel<<<blocks, threads>>>(d_a, d_c, n4, iters);
              break;
            case Op::kWrite:
              WriteKernel<<<blocks, threads>>>(d_c, n4, iters,
                                               make_float4(1.0f, 2.0f, 3.0f, 4.0f));
              break;
            case Op::kCopy:
              CopyKernel<<<blocks, threads>>>(d_a, d_c, n4, iters);
              break;
            case Op::kTriad:
              TriadKernel<<<blocks, threads>>>(d_a, d_b, d_c, n4, iters);
              break;
            case Op::kMemcpyD2D:
              for (int k = 0; k < iters; ++k) {
                CHECK_CUDA(cudaMemcpyAsync(d_c, d_a, size,
                                           cudaMemcpyDeviceToDevice, 0));
              }
              break;
            case Op::kMemset:
              for (int k = 0; k < iters; ++k) {
                CHECK_CUDA(cudaMemsetAsync(d_c, 0, size, 0));
              }
              break;
          }
          CHECK_CUDA_KERNEL();
          times.push_back(timer.StopMs());
        }
        const double ms = bench::Median(times);
        const double gbps = bench::MsToGBps(BytesMoved(op, size) * (size_t)iters, ms);
        const double pct = 100.0 * gbps / theoretical;
        const char* state = cold ? "cold" : "warm";
        std::printf("%-12s %-6s %12zu %10d %12.3f %12.1f %7.1f%%\n",
                    OpName(op), state, size, iters, ms, gbps, pct);
        if (csv) {
          csv << OpName(op) << "," << state << "," << size << "," << iters << ","
              << std::fixed << std::setprecision(6) << ms << "," << gbps << ","
              << pct << "\n";
        }
      }
    }
  }

  if (csv) csv.close();
  CHECK_CUDA(cudaFree(flush_buf));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a));
  return EXIT_SUCCESS;
}
