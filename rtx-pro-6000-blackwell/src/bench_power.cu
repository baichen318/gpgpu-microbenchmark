// Sustained power, clock, and thermal behavior under a full-GPU FP32 load.
#include "common.cuh"

#include <nvml.h>

#include <chrono>
#include <thread>

namespace {

__device__ __forceinline__ float Fma(float a, float b, float c) {
  float d;
  asm volatile("fma.rn.f32 %0, %1, %2, %3;"
               : "=f"(d)
               : "f"(a), "f"(b), "f"(c));
  return d;
}

__global__ void PowerKernel(float* sink, long long iters) {
  float a0 = threadIdx.x * 1.0e-6f + blockIdx.x * 1.0e-7f;
  float a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
  float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
  const float b = 1.000001f;
  const float c = 1.0e-7f;
#pragma unroll 4
  for (long long i = 0; i < iters; ++i) {
    a0 = Fma(a0, b, c);
    a1 = Fma(a1, b, c);
    a2 = Fma(a2, b, c);
    a3 = Fma(a3, b, c);
    a4 = Fma(a4, b, c);
    a5 = Fma(a5, b, c);
    a6 = Fma(a6, b, c);
    a7 = Fma(a7, b, c);
  }
  const float r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (r == -1.0f) *sink = r;
}

struct Sample {
  double t_s = 0.0;
  unsigned int sm_mhz = 0;
  unsigned int mem_mhz = 0;
  unsigned int power_mw = 0;
  unsigned int temp_c = 0;
  unsigned int util_gpu = 0;
  unsigned long long throttle = 0;
};

bool ReadSample(nvmlDevice_t dev, double t_s, Sample* s) {
  s->t_s = t_s;
  if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &s->sm_mhz) != NVML_SUCCESS)
    return false;
  if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &s->mem_mhz) != NVML_SUCCESS)
    return false;
  if (nvmlDeviceGetPowerUsage(dev, &s->power_mw) != NVML_SUCCESS) return false;
  if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &s->temp_c) !=
      NVML_SUCCESS)
    return false;
  nvmlUtilization_t util{};
  if (nvmlDeviceGetUtilizationRates(dev, &util) != NVML_SUCCESS) return false;
  s->util_gpu = util.gpu;
  if (nvmlDeviceGetCurrentClocksThrottleReasons(dev, &s->throttle) !=
      NVML_SUCCESS)
    s->throttle = 0;
  return true;
}

const char* ThrottleName(unsigned long long mask) {
  if (mask & nvmlClocksThrottleReasonSwPowerCap) return "sw_power_cap";
  if (mask & nvmlClocksThrottleReasonHwSlowdown) return "hw_slowdown";
  if (mask & nvmlClocksThrottleReasonSwThermalSlowdown) return "sw_thermal";
  if (mask & nvmlClocksThrottleReasonHwThermalSlowdown) return "hw_thermal";
  if (mask & nvmlClocksThrottleReasonHwPowerBrakeSlowdown) return "hw_power_brake";
  if (mask & nvmlClocksThrottleReasonGpuIdle) return "gpu_idle";
  return "none";
}

}  // namespace

int main(int argc, char** argv) {
  const int device = bench::GetIntArg(argc, argv, "--device", 0);
  const std::string out = bench::GetArg(argc, argv, "--out", "");
  long long iters = 400000000ll;
  if (bench::HasArg(argc, argv, "--iters")) {
    iters = std::stoll(bench::GetArg(argc, argv, "--iters", "400000000"));
  }

  CHECK_CUDA(cudaSetDevice(device));
  const bench::DeviceInfo info = bench::GetDeviceInfo(device);
  const int sms = info.prop.multiProcessorCount;
  std::printf("GPU: %s, SMs=%d, iters=%lld\n", info.prop.name, sms, iters);

  nvmlReturn_t nvret = nvmlInit_v2();
  nvmlDevice_t nvdev{};
  if (nvret == NVML_SUCCESS) {
    nvret = nvmlDeviceGetHandleByIndex_v2(device, &nvdev);
  }
  if (nvret != NVML_SUCCESS) {
    std::fprintf(stderr, "NVML initialization failed: %s\n",
                 nvmlErrorString(nvret));
    return EXIT_FAILURE;
  }

  Sample idle;
  ReadSample(nvdev, 0.0, &idle);
  std::printf("Idle: SM=%u MHz, MEM=%u MHz, power=%.1f W, temp=%u C\n",
              idle.sm_mhz, idle.mem_mhz, idle.power_mw / 1000.0, idle.temp_c);

  float* d_sink = nullptr;
  CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float)));
  cudaEvent_t done = nullptr;
  CHECK_CUDA(cudaEventCreate(&done));

  std::ofstream csv;
  if (!out.empty()) {
    csv.open(out);
    if (!csv) {
      std::fprintf(stderr, "Cannot open %s\n", out.c_str());
      return EXIT_FAILURE;
    }
    csv << "elapsed_s,sm_clock_mhz,mem_clock_mhz,power_w,temp_c,gpu_util_pct,"
           "throttle_mask,throttle_reason\n";
  }

  const int blocks = sms * 3;
  const int threads = 512;
  const auto start = std::chrono::steady_clock::now();
  PowerKernel<<<blocks, threads>>>(d_sink, iters);
  CHECK_CUDA_KERNEL();
  CHECK_CUDA(cudaEventRecord(done));

  std::vector<Sample> samples;
  while (cudaEventQuery(done) == cudaErrorNotReady) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const double t =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    Sample s;
    if (ReadSample(nvdev, t, &s)) {
      samples.push_back(s);
      if (csv) {
        csv << std::fixed << std::setprecision(3) << s.t_s << "," << s.sm_mhz
            << "," << s.mem_mhz << "," << std::setprecision(2)
            << s.power_mw / 1000.0 << "," << s.temp_c << "," << s.util_gpu
            << "," << s.throttle << "," << ThrottleName(s.throttle) << "\n";
      }
    }
  }
  CHECK_CUDA(cudaEventSynchronize(done));
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();

  if (csv) csv.close();
  nvmlShutdown();

  if (samples.empty()) {
    std::printf("No NVML samples collected.\n");
  } else {
    std::vector<double> sm, mem, power, temp;
    int power_capped = 0;
    for (const Sample& s : samples) {
      sm.push_back(s.sm_mhz);
      mem.push_back(s.mem_mhz);
      power.push_back(s.power_mw / 1000.0);
      temp.push_back(s.temp_c);
      if (s.throttle & nvmlClocksThrottleReasonSwPowerCap) ++power_capped;
    }
    std::printf("Sustained %.2f s, %zu samples\n", elapsed, samples.size());
    std::printf("SM clock: min=%.0f avg=%.0f max=%.0f MHz\n", bench::Min(sm),
                bench::Mean(sm), bench::Max(sm));
    std::printf("MEM clock: min=%.0f avg=%.0f max=%.0f MHz\n", bench::Min(mem),
                bench::Mean(mem), bench::Max(mem));
    std::printf("Power: min=%.1f avg=%.1f max=%.1f W\n", bench::Min(power),
                bench::Mean(power), bench::Max(power));
    std::printf("Temperature: min=%.0f avg=%.0f max=%.0f C\n", bench::Min(temp),
                bench::Mean(temp), bench::Max(temp));
    std::printf("Samples with SW power cap: %d/%zu\n", power_capped,
                samples.size());
  }

  CHECK_CUDA(cudaEventDestroy(done));
  CHECK_CUDA(cudaFree(d_sink));
  return EXIT_SUCCESS;
}
