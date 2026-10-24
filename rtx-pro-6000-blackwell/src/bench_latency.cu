// Pointer-chasing latency across the Blackwell memory hierarchy.
#include "common.cuh"

#include <cinttypes>
#include <random>

namespace {

constexpr size_t kCacheLine = 64;

__global__ void BuildRingKernel(unsigned long long* ring, size_t nodes,
                                size_t stride) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = (size_t)gridDim.x * blockDim.x;
  for (; i < nodes; i += step) {
    // One node per 64-byte cache line.  The pointer is stored in the first
    // 8 bytes of the line, exactly like the ring in the reference report.
    ring[i * (kCacheLine / sizeof(unsigned long long))] =
        (unsigned long long)((i + stride) & (nodes - 1));
  }
}

__global__ void ChaseKernel(const unsigned long long* ring,
                            unsigned long long start, unsigned long long iters,
                            int bypass_l1, unsigned long long* cycles_out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  unsigned long long idx = start;
  const unsigned long long t0 = bench::Clock64();
#pragma unroll 1
  for (unsigned long long i = 0; i < iters; ++i) {
    const unsigned long long* p =
        ring + idx * (kCacheLine / sizeof(unsigned long long));
    idx = bypass_l1 ? bench::LoadCgU64(p) : bench::LoadCaU64(p);
  }
  const unsigned long long t1 = bench::Clock64();
  cycles_out[0] = t1 - t0;
  // Consume the final index so the compiler cannot delete the dependent-load
  // loop.  Without this, nvcc's optimizer removes the whole loop because the
  // result is otherwise unused.
  cycles_out[1] = idx;
}

__global__ void ClockWarmupKernel(float* sink, int iters) {
  float a = threadIdx.x * 0.001f + blockIdx.x * 0.0001f;
  float b = 1.000001f;
  for (int i = 0; i < iters; ++i) {
    a = fmaf(a, b, 0.5f);
  }
  if (a == -1.0f) sink[0] = a;
}

size_t NextPow2(size_t v) {
  size_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

void BuildRing(unsigned long long* d_ring, size_t nodes, int stride_mode,
               cudaStream_t stream) {
  // nodes is a power of two.  An odd stride generates one cycle through all
  // nodes.  "half" jumps by half the buffer; "prime" uses a large odd stride
  // that spreads accesses over L2 sets more uniformly.
  size_t stride = (stride_mode == 0) ? (nodes / 2 + 1) : 8191;
  if (stride >= nodes) stride = nodes / 2 + 1;
  if (stride < 1) stride = 1;
  if ((stride & 1) == 0) stride += 1;
  const int threads = 256;
  int blocks = (int)((nodes + threads - 1) / threads);
  if (blocks > 65535) blocks = 65535;
  BuildRingKernel<<<blocks, threads, 0, stream>>>(d_ring, nodes, stride);
  CHECK_CUDA_KERNEL();
}

double ChaseOnce(unsigned long long* d_ring, size_t nodes, int bypass_l1,
                 unsigned long long iters, cudaStream_t stream,
                 unsigned long long* d_cycles, double* cycles_out) {
  bench::GpuTimer timer;
  timer.Start();
  ChaseKernel<<<1, 1, 0, stream>>>(d_ring, 0, iters, bypass_l1, d_cycles);
  CHECK_CUDA_KERNEL();
  const double ms = timer.StopMs();
  unsigned long long cycles[2] = {0, 0};
  CHECK_CUDA(cudaMemcpyAsync(cycles, d_cycles, sizeof(cycles),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  *cycles_out = (double)cycles[0];
  return ms;
}

}  // namespace

int main(int argc, char** argv) {
  const int device = bench::GetIntArg(argc, argv, "--device", 0);
  const size_t max_size = bench::GetSizeArg(argc, argv, "--max-size", 4ull << 30);
  const int repeat = bench::GetIntArg(argc, argv, "--repeat", 3);
  const std::string out = bench::GetArg(argc, argv, "--out", "");
  const bool both_modes = !bench::HasArg(argc, argv, "--ca-only");
  const std::string stride_mode_name =
      bench::GetArg(argc, argv, "--stride-mode", "half");
  const int stride_mode = (stride_mode_name == "prime") ? 1 : 0;

  CHECK_CUDA(cudaSetDevice(device));
  const bench::DeviceInfo info = bench::GetDeviceInfo(device);
  std::printf("GPU: %s, SMs=%d, L2=%d bytes\n", info.prop.name,
              info.prop.multiProcessorCount, info.prop.l2CacheSize);

  const size_t max_nodes = max_size / kCacheLine;
  unsigned long long* d_ring = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ring, max_nodes * kCacheLine));
  unsigned long long* d_cycles = nullptr;
  CHECK_CUDA(cudaMalloc(&d_cycles, 2 * sizeof(unsigned long long)));

  // A buffer larger than L2 is used to evict the ring between sizes.  The
  // pointer chase itself is what we want to measure, so no flush is performed
  // between the repeated runs of the same size.
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
    csv << "mode,stride_mode,size_bytes,nodes,iterations,latency_ns,"
           "cycles_per_load,effective_clock_mhz,bandwidth_equivalent_gbps\n";
  }

  const size_t sizes[] = {
      4ull << 10,     8ull << 10,     16ull << 10,    32ull << 10,
      64ull << 10,    128ull << 10,   256ull << 10,   512ull << 10,
      1ull << 20,     2ull << 20,     4ull << 20,     8ull << 20,
      16ull << 20,    32ull << 20,    64ull << 20,    128ull << 20,
      256ull << 20,   512ull << 20,   1ull << 30,     2ull << 30,
      4ull << 30,     8ull << 30,
  };

  std::printf("%10s %8s %12s %14s %14s %10s\n", "size", "nodes", "latency_ns",
              "cycles/load", "eff_clock_MHz", "GB/s-eq");

  for (size_t size : sizes) {
    if (size > max_size) continue;
    size_t nodes = NextPow2(size / kCacheLine);
    if (nodes < 2) nodes = 2;
    const size_t bytes = nodes * kCacheLine;
    if (bytes > max_size) continue;

    bench::FlushL2(flush_buf, flush_bytes);
    BuildRing(d_ring, nodes, stride_mode, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Boost the SM clock before the single-thread latency measurement.
    ClockWarmupKernel<<<info.prop.multiProcessorCount * 4, 256>>>(nullptr, 200000);
    CHECK_CUDA_KERNEL();
    CHECK_CUDA(cudaDeviceSynchronize());

    // Walk the whole ring at least once.  A short walk only touches a small
    // fraction of a large working set and makes the cache look larger than it
    // really is.
    unsigned long long iters = (unsigned long long)
        std::max(1000000.0, std::min(20000000.0, (double)nodes * 2.0));

    for (int mode = 0; mode < (both_modes ? 2 : 1); ++mode) {
      std::vector<double> lat;
      std::vector<double> cyc;
      std::vector<double> clk;
      for (int r = 0; r < repeat; ++r) {
        bench::FlushL2(flush_buf, flush_bytes);
        CHECK_CUDA(cudaDeviceSynchronize());
        double cycles = 0.0;
        const double ms =
            ChaseOnce(d_ring, nodes, mode, iters, 0, d_cycles, &cycles);
        const double ns = (ms * 1.0e6) / (double)iters;
        lat.push_back(ns);
        cyc.push_back(cycles / (double)iters);
        clk.push_back(cycles / (ms * 1.0e3));
      }
      const double lat_ns = bench::Median(lat);
      const double cycles_per_load = bench::Median(cyc);
      const double clock_mhz = bench::Median(clk);
      const double gbps_eq = 64.0 / lat_ns;
      const char* mode_name = mode ? "cg" : "ca";
      std::printf("%10zu %8zu %12.3f %14.3f %14.1f %10.3f\n", size, nodes,
                  lat_ns, cycles_per_load, clock_mhz, gbps_eq);
      if (csv) {
        csv << mode_name << "," << stride_mode_name << "," << size << ","
            << nodes << "," << iters << "," << std::fixed
            << std::setprecision(6) << lat_ns << ","
            << cycles_per_load << "," << clock_mhz << "," << gbps_eq << "\n";
      }
    }
  }

  if (csv) csv.close();
  CHECK_CUDA(cudaFree(flush_buf));
  CHECK_CUDA(cudaFree(d_cycles));
  CHECK_CUDA(cudaFree(d_ring));
  return EXIT_SUCCESS;
}
