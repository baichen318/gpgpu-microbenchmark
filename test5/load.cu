// Author: eecbai@ust.hk

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "profiler.h"
#include <nvml.h>


#define KiB 1024
#define MiB (KiB * KiB)
#define GiB (KiB * KiB * KiB)
#define GHz (1000L * 1000L * 1000L)
#define NOUTER_ITERS 1L
#define NINNER_ITERS 50L
#define CACHE_LINE_LENGTH 128L
#define STRIDE_START 1L
#define STRIDE_END 1L
#define ALLOCATION_START (512L)
#define ALLOCATION_END (512L * MiB)
#define MEM_LD_LATENCY
// We do not use `INST_LATENCY` for this test.
// #define INST_LATENCY

#define ENABLE_PROFILING

#define CHECK_CUDA(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
            __FILE__, __LINE__, err, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
  } \
}


#define CHECK_NVML(call) { \
  nvmlReturn_t err = call; \
  if (err != NVML_SUCCESS) { \
      fprintf(stderr, "NVML error at %s:%d code=%d(%s)\n", \
              __FILE__, __LINE__, err, nvmlErrorString(err)); \
      exit(EXIT_FAILURE); \
  } \
}


void get_gpu_clock_speed(int device_id, unsigned int *sm_clock, unsigned int *mem_clock, bool show = false) {
  nvmlDevice_t device;

  // Initialize NVML.
  CHECK_NVML(nvmlInit());

  CHECK_NVML(nvmlDeviceGetHandleByIndex(device_id, &device));

  // Get SM frequence (MHz)
  CHECK_NVML(nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, sm_clock));
  if (show)
    printf("Current SM Clock: %u MHz\n", *sm_clock);

  // Get memory frequency (MHz)
  CHECK_NVML(nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, mem_clock));
  if (show)
    printf("Current Memory Clock: %u MHz\n", *mem_clock);

  nvmlShutdown();
}


__global__ void lat(const size_t ncache_lines, char* P, char* dummy, long long int* cycles) {
  const size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if(gid > 0) {
    return;
  }

  #if defined(MEM_LD_LATENCY)
    char** p0 = (char**)P;
    // Warmup
    for(size_t n = 0; n < ncache_lines; ++n) {
      p0 = (char**)*p0;
    }

    // `clock64` computes the cycle numbers.
    long long int t0 = clock64();

    char** p1 = (char**)P;

    #pragma unroll 64
    for(size_t n = 0; n < ncache_lines * NINNER_ITERS; ++n) {
      p1 = (char**)*p1;
    }

    *dummy = *(char*)p0 + *(char*)p1;
  #elif defined(INST_LATENCY)
    long long int t0 = clock64();
    float a = 0.9999f;
    for(int n = 0; n < NINNER_ITERS; ++n) {
      #if 0
        a += 0.99991f;
        a *= 0.9991f;
        a += a * 0.999991f;
        a = sqrtf(a);
        a /= 0.999991f;
      #endif // if 0
    }

    #if 0
      printf("%.7f\n", a);
    #endif // if 0

    *dummy = (char)a;
  #endif

  *cycles += clock64() - t0;
}


__global__ void make_ring(const size_t ncache_lines, const size_t as, const size_t st, char* P0, char* P1) {
  const size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if(gid > 0) {
    return;
  }

  /*
   * Suppose as = 512B = 4 (`ncache_lines`) x 128B (`CACHE_LINE_LENGTH`),
   * we have the following memory layout:
   * #cache line | Set index | stored pointer address in P0 | stored pointer address in P1
   * 0           | 0         | P1[128]                      | P0[128]
   * 1           | 128       | P1[256]                      | P0[256]
   * 2           | 256       | P1[384]                      | P0[384]
   * 3           | 384       | P1[0]                        | P0[0]
   * 
   * So, if we access P0[0], then the memory access pattern would be:
   * P0[0] => get P1[128] => NVLink => get P0[256] => NVLink => get P1[384] => get P0[0]
   *   ^                                                                          |
   *   |_____________________________  NVLink  ___________________________________|
   */
  // Create a ring of pointers at the cache line granularity.
  for(size_t i = 0; i < ncache_lines; ++i) {
    *(char**)&P0[(i * CACHE_LINE_LENGTH)] = &P1[((i + st) * CACHE_LINE_LENGTH) % as];
  }
  // Create a ring of pointers at the cache line granularity.
  for(size_t i = 0; i < ncache_lines; ++i) {
    *(char**)&P1[(i * CACHE_LINE_LENGTH)] = &P0[((i + st) * CACHE_LINE_LENGTH) % as];
  }
}


int main(int argc, char *argv[]) {
  struct Profile profile;
  struct ProfileEntry* pe = &profile.profiler_entries[0];
  unsigned int sm_clock, mem_clock;
  pe->time = 0.0;

  // Initialize.
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaDeviceEnablePeerAccess(1, 0));
  char* P;
  char* dummy;
  CHECK_CUDA(cudaMalloc((void**)&P, ALLOCATION_END));
  CHECK_CUDA(cudaMalloc((void**)&dummy, 1));
  printf("Allocating %lu MiB...\n", ALLOCATION_END / MiB);

  CHECK_CUDA(cudaSetDevice(1));
  char* P1;
  CHECK_CUDA(cudaMalloc((void**)&P1, ALLOCATION_END));
  CHECK_CUDA(cudaSetDevice(0));

  // Open files.
  FILE* nfp = fopen("/dev/null", "a");
  FILE* fp = fopen("lat.csv", "a");

  long long int* d_cycles;
  long long int* d_cycles_dummy;
  CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(long long int)));
  CHECK_CUDA(cudaMalloc(&d_cycles_dummy, sizeof(long long int)));

  for(size_t st = STRIDE_START; st <= STRIDE_END; ++st) {
    for(size_t as = ALLOCATION_START; as <= ALLOCATION_END; as *= 2L) {

      const size_t ncache_lines = as / CACHE_LINE_LENGTH;

      #if defined(MEM_LD_LATENCY)
        make_ring<<<1,1>>>(ncache_lines, as, st, P, P1);
      #endif

      // Zero the cycles.
      long long int h_cycles = 0;
      CHECK_CUDA(cudaMemcpy(d_cycles, &h_cycles, sizeof(long long int), cudaMemcpyHostToDevice));

      // Perform the test.
      START_PROFILING(&profile);
      for(size_t i = 0; i < NOUTER_ITERS; ++i) {
        lat<<<1,1>>>(ncache_lines, P, dummy, d_cycles);
      }
      cudaDeviceSynchronize();
      STOP_PROFILING(&profile, "p");

      // Bring the cycle count back from the device
      CHECK_CUDA(cudaMemcpy(&h_cycles, d_cycles, sizeof(long long int), cudaMemcpyDeviceToHost));

      #if defined(MEM_LD_LATENCY)
        double loads = (double)NOUTER_ITERS * ncache_lines * NINNER_ITERS;
        double cycles_load = ((double)h_cycles / loads);
        get_gpu_clock_speed(0, &sm_clock, &mem_clock);
        double cycles_s = sm_clock * 1.0E-3; // GHz
        double time_per_load = ((double)h_cycles * (1 / cycles_s)) / loads;
        printf(
          "Array size: %.4f MB, stride: %d, #cache lines: %d, elapsed clock cycles: %lu, " \
          "#loads: %.0f, cycles per load: %.4f, time per load: %.8f ns\n", 
          (double)as / MiB, st, ncache_lines, h_cycles,
          loads, cycles_load, time_per_load
        );
        fprintf(fp, "%d,%lu,%lu,%.4f,%.4f\n", st, as, h_cycles, cycles_load, time_per_load);
      #elif defined(INST_LATENCY)
        size_t ops = NOUTER_ITERS * NINNER_ITERS;
        printf("Ops: %lu\n", ops);
        printf("Cycles / Op %.4f\n", h_cycles / (double)ops);
      #endif

      h_cycles = 0;
      cudaMemcpy(d_cycles, &h_cycles, sizeof(long long int), cudaMemcpyHostToDevice);

      pe->time = 0.0;
    }
  }

  fclose(nfp);
  fclose(fp);
  cudaFree(P);

  return 0;
}
