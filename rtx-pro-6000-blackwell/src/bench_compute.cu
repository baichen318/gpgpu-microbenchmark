// FP32/FP64/INT32/SFU/FP16 pipeline throughput and FMA latency.
#include "common.cuh"

#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ float FmaF32(float a, float b, float c) {
  float d;
  asm volatile("fma.rn.f32 %0, %1, %2, %3;"
               : "=f"(d)
               : "f"(a), "f"(b), "f"(c));
  return d;
}

__device__ __forceinline__ double FmaF64(double a, double b, double c) {
  double d;
  asm volatile("fma.rn.f64 %0, %1, %2, %3;"
               : "=d"(d)
               : "d"(a), "d"(b), "d"(c));
  return d;
}

__device__ __forceinline__ int ImadI32(int a, int b, int c) {
  int d;
  asm volatile("mad.lo.s32 %0, %1, %2, %3;"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ float Ex2F32(float a) {
  float d;
  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}

__global__ void Fp32PeakKernel(float* sink, int iters) {
  float a0 = threadIdx.x * 1.0e-6f + blockIdx.x * 1.0e-7f;
  float a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
  float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
  const float b = 1.000001f;
  const float c = 1.0e-7f;
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    a0 = FmaF32(a0, b, c);
    a1 = FmaF32(a1, b, c);
    a2 = FmaF32(a2, b, c);
    a3 = FmaF32(a3, b, c);
    a4 = FmaF32(a4, b, c);
    a5 = FmaF32(a5, b, c);
    a6 = FmaF32(a6, b, c);
    a7 = FmaF32(a7, b, c);
  }
  const float r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (r == -1.0f) *sink = r;
}

__global__ void Fp64PeakKernel(double* sink, int iters) {
  double a0 = threadIdx.x * 1.0e-6;
  double a1 = a0 + 0.1, a2 = a0 + 0.2, a3 = a0 + 0.3;
  double a4 = a0 + 0.4, a5 = a0 + 0.5, a6 = a0 + 0.6, a7 = a0 + 0.7;
  const double b = 1.0000001;
  const double c = 1.0e-9;
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    a0 = FmaF64(a0, b, c);
    a1 = FmaF64(a1, b, c);
    a2 = FmaF64(a2, b, c);
    a3 = FmaF64(a3, b, c);
    a4 = FmaF64(a4, b, c);
    a5 = FmaF64(a5, b, c);
    a6 = FmaF64(a6, b, c);
    a7 = FmaF64(a7, b, c);
  }
  const double r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (r == -1.0) *sink = r;
}

__global__ void Int32PeakKernel(int* sink, int iters) {
  int a0 = threadIdx.x + 1, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
  int a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  const int b = 3;
  const int c = 7;
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    a0 = ImadI32(a0, b, c);
    a1 = ImadI32(a1, b, c);
    a2 = ImadI32(a2, b, c);
    a3 = ImadI32(a3, b, c);
    a4 = ImadI32(a4, b, c);
    a5 = ImadI32(a5, b, c);
    a6 = ImadI32(a6, b, c);
    a7 = ImadI32(a7, b, c);
  }
  const int r = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
  if (r == -1) *sink = r;
}

__global__ void SfuPeakKernel(float* sink, int iters) {
  float a0 = threadIdx.x * 1.0e-6f + 1.0f;
  float a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
  float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    a0 = Ex2F32(a0);
    a1 = Ex2F32(a1);
    a2 = Ex2F32(a2);
    a3 = Ex2F32(a3);
    a4 = Ex2F32(a4);
    a5 = Ex2F32(a5);
    a6 = Ex2F32(a6);
    a7 = Ex2F32(a7);
  }
  const float r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (r == -1.0f) *sink = r;
}

__global__ void Fp16PeakKernel(__half* sink, int iters) {
  __half2 a0 = __float2half2_rn(1.0f + threadIdx.x * 1.0e-6f);
  __half2 a1 = __float2half2_rn(1.1f), a2 = __float2half2_rn(1.2f);
  __half2 a3 = __float2half2_rn(1.3f), a4 = __float2half2_rn(1.4f);
  __half2 a5 = __float2half2_rn(1.5f), a6 = __float2half2_rn(1.6f);
  __half2 a7 = __float2half2_rn(1.7f);
  const __half2 b = __float2half2_rn(1.0001f);
  const __half2 c = __float2half2_rn(0.0001f);
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    a0 = __hfma2(a0, b, c);
    a1 = __hfma2(a1, b, c);
    a2 = __hfma2(a2, b, c);
    a3 = __hfma2(a3, b, c);
    a4 = __hfma2(a4, b, c);
    a5 = __hfma2(a5, b, c);
    a6 = __hfma2(a6, b, c);
    a7 = __hfma2(a7, b, c);
  }
  const __half2 r = __hadd2(__hadd2(__hadd2(a0, a1), __hadd2(a2, a3)),
                            __hadd2(__hadd2(a4, a5), __hadd2(a6, a7)));
  if (__low2float(r) == -1.0f) *sink = __low2half(r);
}

__global__ void Fp32ChainKernel(unsigned long long* out, int iters) {
  if (threadIdx.x != 0) return;
  float a = 1.0f;
  const float b = 1.000001f;
  const float c = 1.0e-7f;
  const unsigned long long t0 = bench::Clock64();
#pragma unroll 16
  for (int i = 0; i < iters; ++i) a = FmaF32(a, b, c);
  const unsigned long long t1 = bench::Clock64();
  out[blockIdx.x * 2 + 0] = t1 - t0;
  out[blockIdx.x * 2 + 1] = (unsigned long long)__float_as_uint(a);
}

__global__ void Fp32IndependentKernel(unsigned long long* out, int iters) {
  if (threadIdx.x != 0) return;
  float a0 = 1.0f, a1 = 1.1f, a2 = 1.2f, a3 = 1.3f;
  float a4 = 1.4f, a5 = 1.5f, a6 = 1.6f, a7 = 1.7f;
  const float b = 1.000001f;
  const float c = 1.0e-7f;
  const unsigned long long t0 = bench::Clock64();
#pragma unroll 16
  for (int i = 0; i < iters; ++i) {
    a0 = FmaF32(a0, b, c);
    a1 = FmaF32(a1, b, c);
    a2 = FmaF32(a2, b, c);
    a3 = FmaF32(a3, b, c);
    a4 = FmaF32(a4, b, c);
    a5 = FmaF32(a5, b, c);
    a6 = FmaF32(a6, b, c);
    a7 = FmaF32(a7, b, c);
  }
  const unsigned long long t1 = bench::Clock64();
  out[blockIdx.x * 2 + 0] = t1 - t0;
  out[blockIdx.x * 2 + 1] =
      (unsigned long long)__float_as_uint(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
}

}  // namespace

int main(int argc, char** argv) {
  const int device = bench::GetIntArg(argc, argv, "--device", 0);
  const int repeat = bench::GetIntArg(argc, argv, "--repeat", 3);
  const std::string out = bench::GetArg(argc, argv, "--out", "");

  CHECK_CUDA(cudaSetDevice(device));
  const bench::DeviceInfo info = bench::GetDeviceInfo(device);
  const int sms = info.prop.multiProcessorCount;
  const int threads = 512;
  const int blocks = sms * 3;
  const double clock_ghz = (double)info.sm_clock_khz / 1.0e6;
  std::printf("GPU: %s, SMs=%d, max SM clock=%.3f GHz\n", info.prop.name, sms,
              clock_ghz);

  float* d_f = nullptr;
  double* d_d = nullptr;
  int* d_i = nullptr;
  __half* d_h = nullptr;
  unsigned long long* d_cycles = nullptr;
  CHECK_CUDA(cudaMalloc(&d_f, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_d, sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_i, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_h, sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_cycles, 2 * (size_t)sms * sizeof(unsigned long long)));

  std::ofstream csv;
  if (!out.empty()) {
    csv.open(out);
    if (!csv) {
      std::fprintf(stderr, "Cannot open %s\n", out.c_str());
      return EXIT_FAILURE;
    }
    csv << "metric,value,unit,note\n";
  }

  // Boost the GPU clock before the short measurements.  A single 0.1 ms kernel
  // is not enough to move the GPU out of its idle clock state.
  for (int w = 0; w < 10; ++w) {
    Fp32PeakKernel<<<blocks, threads>>>(d_f, 1000000);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  auto run_peak = [&](const char* name, auto launcher, int iters,
                      double flops_per_thread_iter, const char* unit,
                      double unit_scale) {
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
      bench::GpuTimer timer;
      timer.Start();
      launcher();
      CHECK_CUDA_KERNEL();
      times.push_back(timer.StopMs());
    }
    const double ms = bench::Median(times);
    const double ops = (double)blocks * threads * iters * flops_per_thread_iter;
    const double value = ops / (ms * unit_scale);
    std::printf("%-16s %10.3f %s (%.3f ms)\n", name, value, unit, ms);
    if (csv) {
      csv << name << "," << std::fixed << std::setprecision(6) << value << ","
          << unit << ",blocks=" << blocks << ";threads=" << threads
          << ";iters=" << iters << "\n";
    }
  };

  // FP32: 8 independent FMAs per thread per iteration, 2 FLOPs per FMA.
  run_peak("fp32_fma", [&] { Fp32PeakKernel<<<blocks, threads>>>(d_f, 20000); },
           20000, 8.0 * 2.0, "TFLOPS", 1.0e9);
  // FP64: 8 independent FMAs per thread per iteration.
  run_peak("fp64_fma", [&] { Fp64PeakKernel<<<blocks, threads>>>(d_d, 500); },
           500, 8.0 * 2.0, "TFLOPS", 1.0e9);
  // INT32: 8 IMADs per thread per iteration, 1 integer op each.
  run_peak("int32_imad", [&] { Int32PeakKernel<<<blocks, threads>>>(d_i, 20000); },
           20000, 8.0, "GIOPS", 1.0e6);
  // SFU: 8 ex2.approx.f32 per thread per iteration.
  run_peak("sfu_ex2", [&] { SfuPeakKernel<<<blocks, threads>>>(d_f, 5000); },
           5000, 8.0, "GIOPS", 1.0e6);
  // FP16: 8 HFMA2 per thread per iteration, each HFMA2 is 2 FMAs (4 FLOPs).
  run_peak("fp16_hfma2", [&] { Fp16PeakKernel<<<blocks, threads>>>(d_h, 10000); },
           10000, 8.0 * 4.0, "TFLOPS", 1.0e9);

  // Dependent-chain and independent-throughput measurements.
  {
    const int iters = 10000000;
    bench::GpuTimer timer;
    timer.Start();
    Fp32ChainKernel<<<sms, 32>>>(d_cycles, iters);
    CHECK_CUDA_KERNEL();
    const double ms = timer.StopMs();
    std::vector<unsigned long long> h(2 * sms, 0);
    CHECK_CUDA(cudaMemcpy(h.data(), d_cycles,
                          h.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    std::vector<double> cycles;
    for (int i = 0; i < sms; ++i) {
      const double c = (double)h[2 * i] / (double)iters;
      if (c > 0.0) cycles.push_back(c);
    }
    const double cyc = bench::Median(cycles);
    std::printf("%-16s %10.3f cycles/FMA (%.3f ms)\n", "fp32_chain", cyc, ms);
    if (csv) csv << "fp32_fma_dependent_latency," << std::fixed
                 << std::setprecision(6) << cyc << ",cycles/FMA,iters=" << iters
                 << "\n";
  }
  {
    const int iters = 5000000;
    bench::GpuTimer timer;
    timer.Start();
    Fp32IndependentKernel<<<sms, 32>>>(d_cycles, iters);
    CHECK_CUDA_KERNEL();
    const double ms = timer.StopMs();
    std::vector<unsigned long long> h(2 * sms, 0);
    CHECK_CUDA(cudaMemcpy(h.data(), d_cycles,
                          h.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    std::vector<double> cycles;
    for (int i = 0; i < sms; ++i) {
      const double c = (double)h[2 * i] / ((double)iters * 8.0);
      if (c > 0.0) cycles.push_back(c);
    }
    const double cyc = bench::Median(cycles);
    std::printf("%-16s %10.3f cycles/FMA (%.3f ms)\n", "fp32_independent",
                cyc, ms);
    if (csv) csv << "fp32_fma_independent_throughput," << std::fixed
                 << std::setprecision(6) << cyc << ",cycles/FMA,iters=" << iters
                 << "\n";
  }

  // Per-SM per-clock FP32 rate.
  // The peak TFLOPS is re-read from the measured value would require keeping
  // it; instead report the theoretical issue width implied by the device.
  std::printf("\nTheoretical FP32 lanes: 128 per SM (4 partitions x 32)\n");
  std::printf("Theoretical FP32 peak at %.3f GHz: %.2f TFLOPS\n", clock_ghz,
              (double)sms * 128.0 * 2.0 * clock_ghz / 1.0e3);

  if (csv) csv.close();
  CHECK_CUDA(cudaFree(d_cycles));
  CHECK_CUDA(cudaFree(d_h));
  CHECK_CUDA(cudaFree(d_i));
  CHECK_CUDA(cudaFree(d_d));
  CHECK_CUDA(cudaFree(d_f));
  return EXIT_SUCCESS;
}
