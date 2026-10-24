// Common utilities for the RTX PRO 6000 Blackwell microbenchmark suite.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t _err = (call);                                                  \
    if (_err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__,         \
                   __LINE__, cudaGetErrorString(_err), (int)_err);              \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#define CHECK_CUDA_KERNEL()                                                     \
  do {                                                                          \
    cudaError_t _err = cudaGetLastError();                                      \
    if (_err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA kernel error at %s:%d: %s (%d)\n", __FILE__,   \
                   __LINE__, cudaGetErrorString(_err), (int)_err);              \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

namespace bench {

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

class GpuTimer {
 public:
  GpuTimer() {
    CHECK_CUDA(cudaEventCreate(&start_));
    CHECK_CUDA(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void Start() { CHECK_CUDA(cudaEventRecord(start_)); }
  double StopMs() {
    CHECK_CUDA(cudaEventRecord(stop_));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
    return (double)ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

inline double Median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n & 1) return values[n / 2];
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

inline double Min(const std::vector<double>& values) {
  return values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
}

inline double Max(const std::vector<double>& values) {
  return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
}

inline double Mean(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  double sum = 0.0;
  for (double v : values) sum += v;
  return sum / values.size();
}

inline double Stddev(const std::vector<double>& values) {
  if (values.size() < 2) return 0.0;
  const double mean = Mean(values);
  double sum = 0.0;
  for (double v : values) sum += (v - mean) * (v - mean);
  return std::sqrt(sum / (values.size() - 1));
}

inline double MsToGbps(size_t bytes, double ms) {
  if (ms <= 0.0) return 0.0;
  return (double)bytes / (ms * 1.0e6);
}

inline double MsToGBps(size_t bytes, double ms) {
  if (ms <= 0.0) return 0.0;
  return (double)bytes / (ms * 1.0e6);
}

// ---------------------------------------------------------------------------
// Device information
// ---------------------------------------------------------------------------

struct DeviceInfo {
  cudaDeviceProp prop{};
  int sm_clock_khz = 0;
  int memory_clock_khz = 0;
  int cooperative_launch = 0;
  int cluster_launch = 0;
  int max_cluster_size = 0;
  int max_blocks_per_cluster = 0;
  int max_persisting_l2_size = 0;
  int max_access_policy_window = 0;
  int reserved_shared_memory_per_block = 0;
};

inline int GetAttr(int device, cudaDeviceAttr attr) {
  int value = 0;
  cudaError_t err = cudaDeviceGetAttribute(&value, attr, device);
  if (err != cudaSuccess) return -1;
  return value;
}

inline DeviceInfo GetDeviceInfo(int device) {
  DeviceInfo info;
  CHECK_CUDA(cudaGetDeviceProperties(&info.prop, device));
  info.sm_clock_khz = GetAttr(device, cudaDevAttrClockRate);
  info.memory_clock_khz = GetAttr(device, cudaDevAttrMemoryClockRate);
  info.cooperative_launch = GetAttr(device, cudaDevAttrCooperativeLaunch);
  info.cluster_launch = GetAttr(device, cudaDevAttrClusterLaunch);
  info.max_cluster_size = 0;
  info.max_blocks_per_cluster = 0;
  info.max_persisting_l2_size = GetAttr(device, cudaDevAttrMaxPersistingL2CacheSize);
  info.max_access_policy_window = GetAttr(device, cudaDevAttrMaxAccessPolicyWindowSize);
  info.reserved_shared_memory_per_block =
      GetAttr(device, cudaDevAttrReservedSharedMemoryPerBlock);
  return info;
}

inline double TheoreticalHbmBandwidthGbps(const DeviceInfo& info) {
  // NVIDIA reports the memory clock in kHz. For GDDR6/GDDR7 the effective
  // data rate is double data rate (DDR), so the peak is 2 * clock * bus width.
  return 2.0 * (double)info.memory_clock_khz * 1000.0 *
         ((double)info.prop.memoryBusWidth / 8.0) / 1.0e9;
}

inline int MaxPotentialClusterSize(const void* func, int block_size) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1024, 1, 1);
  config.blockDim = dim3(block_size, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = 0;
  int cluster_size = 0;
  cudaError_t err = cudaOccupancyMaxPotentialClusterSize(&cluster_size, func, &config);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return cluster_size;
}

inline void PrintDeviceInfo(const DeviceInfo& info) {
  const cudaDeviceProp& p = info.prop;
  int runtime_version = 0;
  int driver_version = 0;
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
  CHECK_CUDA(cudaDriverGetVersion(&driver_version));

  std::printf("name: %s\n", p.name);
  std::printf("compute_capability: %d.%d\n", p.major, p.minor);
  std::printf("multi_processor_count: %d\n", p.multiProcessorCount);
  std::printf("sm_clock_khz: %d\n", info.sm_clock_khz);
  std::printf("memory_clock_khz: %d\n", info.memory_clock_khz);
  std::printf("memory_bus_width_bits: %d\n", p.memoryBusWidth);
  std::printf("memory_total_bytes: %zu\n", (size_t)p.totalGlobalMem);
  std::printf("memory_free_bytes: %zu\n", (size_t)p.totalGlobalMem);
  std::printf("l2_cache_bytes: %d\n", p.l2CacheSize);
  std::printf("shared_mem_per_sm_bytes: %zu\n", (size_t)p.sharedMemPerMultiprocessor);
  std::printf("shared_mem_per_block_bytes: %zu\n", (size_t)p.sharedMemPerBlock);
  std::printf("shared_mem_per_block_optin_bytes: %zu\n", (size_t)p.sharedMemPerBlockOptin);
  std::printf("regs_per_sm: %d\n", p.regsPerMultiprocessor);
  std::printf("regs_per_block: %d\n", p.regsPerBlock);
  std::printf("max_threads_per_sm: %d\n", p.maxThreadsPerMultiProcessor);
  std::printf("max_threads_per_block: %d\n", p.maxThreadsPerBlock);
  std::printf("max_blocks_per_sm: %d\n", p.maxBlocksPerMultiProcessor);
  std::printf("warp_size: %d\n", p.warpSize);
  std::printf("async_engine_count: %d\n", p.asyncEngineCount);
  std::printf("concurrent_kernels: %d\n", p.concurrentKernels);
  std::printf("unified_addressing: %d\n", p.unifiedAddressing);
  std::printf("managed_memory: %d\n", p.managedMemory);
  std::printf("pageable_memory_access: %d\n", p.pageableMemoryAccess);
  std::printf("can_map_host_memory: %d\n", p.canMapHostMemory);
  std::printf("pci_bus_id: %d\n", p.pciBusID);
  std::printf("pci_device_id: %d\n", p.pciDeviceID);
  std::printf("pci_domain_id: %d\n", p.pciDomainID);
  std::printf("theoretical_hbm_bandwidth_gbps: %.3f\n",
              TheoreticalHbmBandwidthGbps(info));
  std::printf("cooperative_launch: %d\n", info.cooperative_launch);
  std::printf("cluster_launch: %d\n", info.cluster_launch);
  std::printf("max_cluster_size: %d\n", info.max_cluster_size);
  std::printf("max_blocks_per_cluster: %d\n", info.max_blocks_per_cluster);
  std::printf("max_persisting_l2_size_bytes: %d\n", info.max_persisting_l2_size);
  std::printf("max_access_policy_window_bytes: %d\n", info.max_access_policy_window);
  std::printf("reserved_shared_memory_per_block_bytes: %d\n",
              info.reserved_shared_memory_per_block);
  std::printf("cuda_runtime_version: %d\n", runtime_version);
  std::printf("cuda_driver_version: %d\n", driver_version);
}

inline void WriteDeviceInfoCsv(const DeviceInfo& info, std::ostream& os) {
  const cudaDeviceProp& p = info.prop;
  int runtime_version = 0;
  int driver_version = 0;
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
  CHECK_CUDA(cudaDriverGetVersion(&driver_version));
  os << "key,value,unit\n";
  auto row = [&](const char* key, double value, const char* unit) {
    os << key << "," << std::setprecision(12) << value << "," << unit << "\n";
  };
  os << "name,\"" << p.name << "\",string\n";
  row("compute_capability_major", p.major, "version");
  row("compute_capability_minor", p.minor, "version");
  row("multi_processor_count", p.multiProcessorCount, "SMs");
  row("sm_clock_khz", info.sm_clock_khz, "kHz");
  row("memory_clock_khz", info.memory_clock_khz, "kHz");
  row("memory_bus_width_bits", p.memoryBusWidth, "bits");
  row("memory_total_bytes", (double)p.totalGlobalMem, "bytes");
  row("l2_cache_bytes", p.l2CacheSize, "bytes");
  row("shared_mem_per_sm_bytes", (double)p.sharedMemPerMultiprocessor, "bytes");
  row("shared_mem_per_block_bytes", (double)p.sharedMemPerBlock, "bytes");
  row("shared_mem_per_block_optin_bytes", (double)p.sharedMemPerBlockOptin, "bytes");
  row("regs_per_sm", p.regsPerMultiprocessor, "registers");
  row("regs_per_block", p.regsPerBlock, "registers");
  row("max_threads_per_sm", p.maxThreadsPerMultiProcessor, "threads");
  row("max_threads_per_block", p.maxThreadsPerBlock, "threads");
  row("max_blocks_per_sm", p.maxBlocksPerMultiProcessor, "blocks");
  row("warp_size", p.warpSize, "threads");
  row("async_engine_count", p.asyncEngineCount, "engines");
  row("concurrent_kernels", p.concurrentKernels, "bool");
  row("unified_addressing", p.unifiedAddressing, "bool");
  row("managed_memory", p.managedMemory, "bool");
  row("pageable_memory_access", p.pageableMemoryAccess, "bool");
  row("can_map_host_memory", p.canMapHostMemory, "bool");
  row("pci_bus_id", p.pciBusID, "id");
  row("pci_device_id", p.pciDeviceID, "id");
  row("pci_domain_id", p.pciDomainID, "id");
  row("theoretical_hbm_bandwidth_gbps", TheoreticalHbmBandwidthGbps(info), "GB/s");
  row("cooperative_launch", info.cooperative_launch, "bool");
  row("cluster_launch", info.cluster_launch, "bool");
  row("max_cluster_size", info.max_cluster_size, "blocks");
  row("max_blocks_per_cluster", info.max_blocks_per_cluster, "blocks");
  row("max_persisting_l2_size_bytes", info.max_persisting_l2_size, "bytes");
  row("max_access_policy_window_bytes", info.max_access_policy_window, "bytes");
  row("reserved_shared_memory_per_block_bytes",
      info.reserved_shared_memory_per_block, "bytes");
  row("cuda_runtime_version", runtime_version, "version");
  row("cuda_driver_version", driver_version, "version");
}

// ---------------------------------------------------------------------------
// Cache control
// ---------------------------------------------------------------------------

__global__ void FlushL2Kernel(int4* data, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  int4 v = make_int4(1, 1, 1, 1);
  for (; i < n4; i += stride) {
    data[i] = v;
  }
}

inline void FlushL2(void* buffer, size_t bytes, cudaStream_t stream = 0) {
  if (buffer == nullptr || bytes < 16) return;
  const size_t n4 = bytes / sizeof(int4);
  const int threads = 256;
  int blocks = (int)((n4 + threads - 1) / threads);
  if (blocks > 65535) blocks = 65535;
  FlushL2Kernel<<<blocks, threads, 0, stream>>>((int4*)buffer, n4);
  CHECK_CUDA_KERNEL();
}

// ---------------------------------------------------------------------------
// Device-side helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int SmId() {
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

__device__ __forceinline__ unsigned long long GlobalTimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ unsigned long long Clock64() {
  return (unsigned long long)clock64();
}

__device__ __forceinline__ unsigned long long LoadCaU64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long LoadCgU64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ float4 LoadGlobalFloat4(const float4* p) {
  float4 v;
  asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ void StoreGlobalFloat4(float4* p, float4 v) {
  asm volatile("st.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
               : "memory");
}

__device__ __forceinline__ float LoadSharedFloat(const float* p) {
  float v;
  const unsigned int addr = (unsigned int)__cvta_generic_to_shared(p);
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr) : "memory");
  return v;
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

inline bool HasArg(int argc, char** argv, const std::string& name) {
  for (int i = 1; i < argc; ++i) {
    if (name == argv[i]) return true;
  }
  return false;
}

inline std::string GetArg(int argc, char** argv, const std::string& name,
                          const std::string& default_value) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) return argv[i + 1];
  }
  return default_value;
}

inline int GetIntArg(int argc, char** argv, const std::string& name, int default_value) {
  return std::atoi(GetArg(argc, argv, name, std::to_string(default_value)).c_str());
}

inline size_t ParseSize(const std::string& s) {
  if (s.empty()) return 0;
  char* end = nullptr;
  double value = std::strtod(s.c_str(), &end);
  if (end == s.c_str()) return 0;
  std::string suffix(end);
  while (!suffix.empty() && std::isspace((unsigned char)suffix[0])) suffix.erase(0, 1);
  for (char& c : suffix) c = (char)std::tolower((unsigned char)c);
  double mul = 1.0;
  if (suffix == "k" || suffix == "kb") mul = 1024.0;
  else if (suffix == "m" || suffix == "mb") mul = 1024.0 * 1024.0;
  else if (suffix == "g" || suffix == "gb") mul = 1024.0 * 1024.0 * 1024.0;
  else if (suffix == "t" || suffix == "tb") mul = 1024.0 * 1024.0 * 1024.0 * 1024.0;
  return (size_t)(value * mul);
}

inline size_t GetSizeArg(int argc, char** argv, const std::string& name,
                         size_t default_value) {
  return ParseSize(GetArg(argc, argv, name, std::to_string(default_value)));
}

inline int DefaultBlocks(const cudaDeviceProp& prop, int multiplier = 4) {
  return prop.multiProcessorCount * multiplier;
}

inline void PrintHeader(const char* title) {
  std::printf("============================================================\n");
  std::printf("%s\n", title);
  std::printf("============================================================\n");
}

}  // namespace bench
