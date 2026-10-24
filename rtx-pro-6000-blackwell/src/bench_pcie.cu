// PCIe host-device bandwidth and small-transfer latency.
#include "common.cuh"

namespace {

double MedianTime(std::vector<double> values) { return bench::Median(values); }

}  // namespace

int main(int argc, char** argv) {
  const int device = bench::GetIntArg(argc, argv, "--device", 0);
  const int repeat = bench::GetIntArg(argc, argv, "--repeat", 10);
  const size_t max_size = bench::GetSizeArg(argc, argv, "--max-size", 1ull << 30);
  const std::string out = bench::GetArg(argc, argv, "--out", "");

  CHECK_CUDA(cudaSetDevice(device));
  const bench::DeviceInfo info = bench::GetDeviceInfo(device);
  std::printf("GPU: %s, PCI bus %04x:%02x:%02x\n", info.prop.name,
              info.prop.pciDomainID, info.prop.pciBusID, info.prop.pciDeviceID);

  void* h_pinned = nullptr;
  void* d_buf = nullptr;
  CHECK_CUDA(cudaHostAlloc(&h_pinned, max_size, cudaHostAllocDefault));
  CHECK_CUDA(cudaMalloc(&d_buf, max_size));
  std::memset(h_pinned, 0x5a, max_size);
  CHECK_CUDA(cudaMemset(d_buf, 0x3c, max_size));
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));

  std::ofstream csv;
  if (!out.empty()) {
    csv.open(out);
    if (!csv) {
      std::fprintf(stderr, "Cannot open %s\n", out.c_str());
      return EXIT_FAILURE;
    }
    csv << "direction,memory_type,size_bytes,time_us,bandwidth_gbps\n";
  }

  const size_t sizes[] = {
      1ull << 10,   4ull << 10,   16ull << 10,  64ull << 10,
      256ull << 10, 1ull << 20,   4ull << 20,   16ull << 20,
      64ull << 20,  256ull << 20, 512ull << 20, 1ull << 30,
  };

  std::printf("%-10s %12s %12s %14s %12s\n", "direction", "size", "time_us",
              "GB/s", "");
  for (size_t size : sizes) {
    if (size > max_size) continue;
    for (int dir = 0; dir < 2; ++dir) {
      const bool h2d = (dir == 0);
      std::vector<double> times;
      const int reps = (size >= (64ull << 20)) ? repeat : std::max(repeat, 20);
      for (int r = 0; r < reps; ++r) {
        bench::GpuTimer timer;
        timer.Start();
        if (h2d) {
          CHECK_CUDA(cudaMemcpyAsync(d_buf, h_pinned, size,
                                     cudaMemcpyHostToDevice, stream));
        } else {
          CHECK_CUDA(cudaMemcpyAsync(h_pinned, d_buf, size,
                                     cudaMemcpyDeviceToHost, stream));
        }
        const double ms = timer.StopMs();
        times.push_back(ms);
      }
      const double ms = MedianTime(times);
      const double us = ms * 1000.0;
      const double gbps = bench::MsToGBps(size, ms);
      std::printf("%-10s %12zu %12.3f %14.3f\n", h2d ? "H2D" : "D2H", size, us,
                  gbps);
      if (csv) {
        csv << (h2d ? "H2D" : "D2H") << ",pinned," << size << "," << std::fixed
            << std::setprecision(6) << us << "," << gbps << "\n";
      }
    }
  }

  // Pageable-memory reference for two representative sizes.
  for (size_t size : {64ull << 20, 256ull << 20}) {
    if (size > max_size) continue;
    void* h_pageable = std::malloc(size);
    std::memset(h_pageable, 0x11, size);
    for (int dir = 0; dir < 2; ++dir) {
      const bool h2d = (dir == 0);
      std::vector<double> times;
      for (int r = 0; r < 5; ++r) {
        bench::GpuTimer timer;
        timer.Start();
        if (h2d) {
          CHECK_CUDA(cudaMemcpy(d_buf, h_pageable, size,
                                cudaMemcpyHostToDevice));
        } else {
          CHECK_CUDA(cudaMemcpy(h_pageable, d_buf, size,
                                cudaMemcpyDeviceToHost));
        }
        times.push_back(timer.StopMs());
      }
      const double ms = MedianTime(times);
      const double gbps = bench::MsToGBps(size, ms);
      std::printf("%-10s %12zu %12.3f %14.3f (pageable)\n",
                  h2d ? "H2D" : "D2H", size, ms * 1000.0, gbps);
      if (csv) {
        csv << (h2d ? "H2D" : "D2H") << ",pageable," << size << ","
            << std::fixed << std::setprecision(6) << ms * 1000.0 << "," << gbps
            << "\n";
      }
    }
    std::free(h_pageable);
  }

  if (csv) csv.close();
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_buf));
  CHECK_CUDA(cudaFreeHost(h_pinned));
  return EXIT_SUCCESS;
}
