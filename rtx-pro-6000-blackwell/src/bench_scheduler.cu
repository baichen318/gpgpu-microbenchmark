// Warp-scheduler issue rate and occupancy limits.
#include "common.cuh"

namespace {

__device__ __forceinline__ int Iadd(int a, int b) {
  int d;
  asm volatile("add.s32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__global__ void IssueKernel(unsigned long long* cycles_out, int iters) {
  int a0 = threadIdx.x + 1, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
  int a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  __syncthreads();
  unsigned long long t0 = 0;
  if (threadIdx.x == 0) t0 = bench::Clock64();
  __syncthreads();
#pragma unroll 2
  for (int i = 0; i < iters; ++i) {
    a0 = Iadd(a0, i + 1);
    a1 = Iadd(a1, i + 2);
    a2 = Iadd(a2, i + 3);
    a3 = Iadd(a3, i + 4);
    a4 = Iadd(a4, i + 5);
    a5 = Iadd(a5, i + 6);
    a6 = Iadd(a6, i + 7);
    a7 = Iadd(a7, i + 8);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long t1 = bench::Clock64();
    cycles_out[blockIdx.x] = t1 - t0;
    cycles_out[blockIdx.x + gridDim.x] =
        (unsigned long long)(a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7);
  }
}

__global__ void SharedOccupancyKernel() {}

}  // namespace

int main(int argc, char** argv) {
  const int device = bench::GetIntArg(argc, argv, "--device", 0);
  const int repeat = bench::GetIntArg(argc, argv, "--repeat", 3);
  const std::string out = bench::GetArg(argc, argv, "--out", "");

  CHECK_CUDA(cudaSetDevice(device));
  const bench::DeviceInfo info = bench::GetDeviceInfo(device);
  const int sms = info.prop.multiProcessorCount;
  const int iters = 100000;
  std::printf("GPU: %s, SMs=%d\n", info.prop.name, sms);

  unsigned long long* d_cycles = nullptr;
  CHECK_CUDA(cudaMalloc(&d_cycles, 2 * (size_t)sms * sizeof(unsigned long long)));

  std::ofstream csv;
  if (!out.empty()) {
    csv.open(out);
    if (!csv) {
      std::fprintf(stderr, "Cannot open %s\n", out.c_str());
      return EXIT_FAILURE;
    }
    csv << "test,threads,warps,cycles,warp_instr_per_cycle\n";
  }

  const int thread_counts[] = {32, 64, 128, 256, 512, 1024};
  std::printf("%10s %8s %14s %20s\n", "threads", "warps", "cycles", "warp-instr/cycle");
  for (int threads : thread_counts) {
    std::vector<double> cycles;
    for (int r = 0; r < repeat; ++r) {
      bench::GpuTimer timer;
      timer.Start();
      IssueKernel<<<sms, threads>>>(d_cycles, iters);
      CHECK_CUDA_KERNEL();
      timer.StopMs();
      std::vector<unsigned long long> h(2 * sms, 0);
      CHECK_CUDA(cudaMemcpy(h.data(), d_cycles,
                            h.size() * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < sms; ++i) {
        if (h[i] > 0) cycles.push_back((double)h[i]);
      }
    }
    const double cyc = bench::Median(cycles);
    const int warps = threads / 32;
    const double warp_instr = (double)warps * 8.0 * (double)iters / cyc;
    std::printf("%10d %8d %14.1f %20.3f\n", threads, warps, cyc, warp_instr);
    if (csv) {
      csv << "issue," << threads << "," << warps << "," << std::fixed
          << std::setprecision(3) << cyc << "," << warp_instr << "\n";
    }
  }

  // Shared-memory occupancy tradeoff.  The empty kernel isolates the effect of
  // the dynamic shared-memory request on blocks/SM.
  CHECK_CUDA(cudaFuncSetAttribute(
      SharedOccupancyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      info.prop.sharedMemPerBlockOptin));
  std::printf("\nOccupancy vs dynamic shared memory (empty kernel):\n");
  std::printf("%14s %14s %12s\n", "shared_bytes", "blocks/SM", "occupancy");
  const size_t shmem_values[] = {0, 16 * 1024, 32 * 1024, 64 * 1024, 96 * 1024};
  for (size_t shmem : shmem_values) {
    if (shmem > (size_t)info.prop.sharedMemPerBlockOptin) continue;
    int blocks = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, SharedOccupancyKernel, 256, shmem);
    if (err != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    const double occ =
        100.0 * (double)blocks * 256.0 / (double)info.prop.maxThreadsPerMultiProcessor;
    std::printf("%14zu %14d %11.1f%%\n", shmem, blocks, occ);
    if (csv) {
      csv << "occupancy," << shmem << ",0,0," << std::fixed
          << std::setprecision(3) << occ << "\n";
    }
  }

  if (csv) csv.close();
  CHECK_CUDA(cudaFree(d_cycles));
  return EXIT_SUCCESS;
}
