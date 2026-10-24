// L1 and shared-memory bandwidth, bank conflicts, and shared-memory latency.
#include "common.cuh"

namespace {

__global__ void L1ReadKernel(const float4* __restrict__ data, size_t n4_per_block,
                             int iters, float4* __restrict__ sink) {
  const float4* base = data + (size_t)blockIdx.x * n4_per_block;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int it = 0; it < iters; ++it) {
    for (size_t i = threadIdx.x; i < n4_per_block; i += blockDim.x) {
      const float4 v = bench::LoadGlobalFloat4(&base[i]);
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
  }
  if (acc.x == -1.0f) *sink = acc;
}

__global__ void L2ReadKernel(const float4* __restrict__ data, size_t n4,
                             int iters, float4* __restrict__ sink) {
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  // Each block walks the whole buffer.  This keeps the per-block working set
  // (16 MiB) larger than L1, so the loads are served by L2.
  const size_t stride = blockDim.x;
  for (int it = 0; it < iters; ++it) {
    for (size_t i = threadIdx.x; i < n4; i += stride) {
      const float4 v = bench::LoadGlobalFloat4(&data[i]);
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
  }
  if (acc.x == -1.0f) *sink = acc;
}

__global__ void SharedReadKernel(int n, int iters, float* sink) {
  extern __shared__ float s[];
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = (float)i * 0.5f;
  __syncthreads();
  float acc = 0.0f;
  for (int it = 0; it < iters; ++it) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      acc += bench::LoadSharedFloat(&s[i]);
  }
  if (acc == -1.0f) *sink = acc;
}

__global__ void SharedBankKernel(int n, int stride, int iters, float* sink) {
  extern __shared__ float s[];
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = (float)i * 0.5f;
  __syncthreads();
  float acc = 0.0f;
  for (int it = 0; it < iters; ++it) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int idx = (i * stride) & (n - 1);
      acc += bench::LoadSharedFloat(&s[idx]);
    }
  }
  if (acc == -1.0f) *sink = acc;
}

__global__ void SharedLatencyKernel(int n, int iters, unsigned int* out) {
  extern __shared__ unsigned int s_lat[];
  const int stride = (n / 2 + 1) | 1;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    s_lat[i] = (unsigned int)((i + stride) & (n - 1));
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned int* vs = s_lat;
    unsigned int idx = 0;
    const unsigned long long t0 = bench::Clock64();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) idx = vs[idx];
    const unsigned long long t1 = bench::Clock64();
    out[blockIdx.x * 2 + 0] = (unsigned int)(t1 - t0);
    out[blockIdx.x * 2 + 1] = idx;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int device = bench::GetIntArg(argc, argv, "--device", 0);
  const int repeat = bench::GetIntArg(argc, argv, "--repeat", 5);
  const std::string out = bench::GetArg(argc, argv, "--out", "");

  CHECK_CUDA(cudaSetDevice(device));
  const bench::DeviceInfo info = bench::GetDeviceInfo(device);
  const int sms = info.prop.multiProcessorCount;
  const int threads = 1024;
  const int l1_bytes_per_block = 64 * 1024;
  const int l1_n4 = l1_bytes_per_block / sizeof(float4);
  const int shared_bytes = 64 * 1024;
  const int shared_n = shared_bytes / sizeof(float);

  std::printf("GPU: %s, SMs=%d, shared/SM=%zu bytes\n", info.prop.name, sms,
              (size_t)info.prop.sharedMemPerMultiprocessor);

  float4* d_l1 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_l1, (size_t)sms * l1_bytes_per_block));
  const size_t l2_bytes = 16ull << 20;
  float4* d_l2 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_l2, l2_bytes));
  CHECK_CUDA(cudaMemset(d_l2, 4, l2_bytes));
  CHECK_CUDA(cudaMemset(d_l1, 3, (size_t)sms * l1_bytes_per_block));
  float4* d_sink = nullptr;
  CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float4)));
  float* d_fsink = nullptr;
  CHECK_CUDA(cudaMalloc(&d_fsink, sizeof(float)));
  unsigned int* d_lat = nullptr;
  CHECK_CUDA(cudaMalloc(&d_lat, 2 * sms * sizeof(unsigned int)));

  CHECK_CUDA(cudaFuncSetAttribute(SharedReadKernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_bytes));
  CHECK_CUDA(cudaFuncSetAttribute(SharedBankKernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_bytes));
  CHECK_CUDA(cudaFuncSetAttribute(SharedLatencyKernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_bytes));

  std::ofstream csv;
  if (!out.empty()) {
    csv.open(out);
    if (!csv) {
      std::fprintf(stderr, "Cannot open %s\n", out.c_str());
      return EXIT_FAILURE;
    }
    csv << "test,config,size_bytes,iterations,time_ms,bandwidth_gbps,"
           "cycles_per_load\n";
  }

  // -------------------- L1 read bandwidth --------------------
  {
    const int iters = 5000;
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
      bench::GpuTimer timer;
      timer.Start();
      L1ReadKernel<<<sms, threads>>>(d_l1, l1_n4, iters, d_sink);
      CHECK_CUDA_KERNEL();
      times.push_back(timer.StopMs());
    }
    const double ms = bench::Median(times);
    const size_t bytes = (size_t)sms * l1_bytes_per_block * (size_t)iters;
    const double gbps = bench::MsToGBps(bytes, ms);
    std::printf("L1 read: %.1f GB/s (%.1f GB/s/SM)\n", gbps, gbps / sms);
    if (csv) {
      csv << "l1_read,per_sm_64KB," << l1_bytes_per_block << "," << iters << ","
          << std::fixed << std::setprecision(6) << ms << "," << gbps << ",0\n";
    }
  }

  // -------------------- L2 read bandwidth --------------------
  {
    const int iters = 10;
    const size_t n4 = l2_bytes / sizeof(float4);
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
      bench::GpuTimer timer;
      timer.Start();
      L2ReadKernel<<<sms, 1024>>>(d_l2, n4, iters, d_sink);
      CHECK_CUDA_KERNEL();
      times.push_back(timer.StopMs());
    }
    const double ms = bench::Median(times);
    const size_t bytes = l2_bytes * (size_t)iters * (size_t)sms;
    const double gbps = bench::MsToGBps(bytes, ms);
    std::printf("L2 read: %.1f GB/s (%.1f GB/s/SM)\n", gbps, gbps / sms);
    if (csv) {
      csv << "l2_read,per_sm_16MB," << l2_bytes << "," << iters << ","
          << std::fixed << std::setprecision(6) << ms << "," << gbps << ",0\n";
    }
  }

  // -------------------- shared-memory read bandwidth --------------------
  {
    const int iters = 5000;
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
      bench::GpuTimer timer;
      timer.Start();
      SharedReadKernel<<<sms, threads, shared_bytes>>>(shared_n, iters, d_fsink);
      CHECK_CUDA_KERNEL();
      times.push_back(timer.StopMs());
    }
    const double ms = bench::Median(times);
    const size_t bytes = (size_t)sms * shared_bytes * (size_t)iters;
    const double gbps = bench::MsToGBps(bytes, ms);
    std::printf("Shared read: %.1f GB/s (%.1f GB/s/SM)\n", gbps, gbps / sms);
    if (csv) {
      csv << "shared_read,stride_1," << shared_bytes << "," << iters << ","
          << std::fixed << std::setprecision(6) << ms << "," << gbps << ",0\n";
    }
  }

  // -------------------- shared-memory bank conflicts --------------------
  const int strides[] = {1, 2, 4, 8, 16, 32};
  std::printf("%8s %12s %12s\n", "stride", "GB/s", "relative");
  double baseline = 0.0;
  for (int stride : strides) {
    const int iters = 2000;
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
      bench::GpuTimer timer;
      timer.Start();
      SharedBankKernel<<<sms, threads, shared_bytes>>>(shared_n, stride, iters,
                                                       d_fsink);
      CHECK_CUDA_KERNEL();
      times.push_back(timer.StopMs());
    }
    const double ms = bench::Median(times);
    const size_t bytes = (size_t)sms * shared_bytes * (size_t)iters;
    const double gbps = bench::MsToGBps(bytes, ms);
    if (stride == 1) baseline = gbps;
    const double rel = baseline > 0.0 ? gbps / baseline : 1.0;
    std::printf("%8d %12.1f %12.3f\n", stride, gbps, rel);
    if (csv) {
      csv << "shared_bank_conflict,stride_" << stride << "," << shared_bytes << ","
          << iters << "," << std::fixed << std::setprecision(6) << ms << ","
          << gbps << ",0\n";
    }
  }

  // -------------------- shared-memory latency --------------------
  {
    const int n = shared_n;
    const int iters = 1000000;
    bench::GpuTimer timer;
    timer.Start();
    SharedLatencyKernel<<<sms, 32, shared_bytes>>>(n, iters, d_lat);
    CHECK_CUDA_KERNEL();
    const double ms = timer.StopMs();
    std::vector<unsigned int> host(2 * sms);
    CHECK_CUDA(cudaMemcpy(host.data(), d_lat, host.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    std::vector<double> cycles;
    for (int i = 0; i < sms; ++i) {
      const double c = (double)host[2 * i] / (double)iters;
      if (c > 0.0) cycles.push_back(c);
    }
    const double cyc = bench::Median(cycles);
    const double ns = cyc * 1.0e6 / (double)info.sm_clock_khz;
    std::printf("Shared latency: %.2f cycles/load (%.2f ns at max clock)\n", cyc,
                ns);
    if (csv) {
      csv << "shared_latency,pointer_chase," << shared_bytes << "," << iters << ","
          << std::fixed << std::setprecision(6) << ms << ",0," << cyc << "\n";
    }
  }

  if (csv) csv.close();
  CHECK_CUDA(cudaFree(d_lat));
  CHECK_CUDA(cudaFree(d_fsink));
  CHECK_CUDA(cudaFree(d_sink));
  CHECK_CUDA(cudaFree(d_l1));
  CHECK_CUDA(cudaFree(d_l2));
  return EXIT_SUCCESS;
}
