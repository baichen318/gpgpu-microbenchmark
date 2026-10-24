// Tensor-core throughput: FP16/BF16 WMMA and FP8/FP4 MMA.
#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace {

using namespace nvcuda;

__global__ void WmmaFp16Kernel(const __half* a, const __half* b, float* c,
                               int iters) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc1, fc2, fc3;
  wmma::load_matrix_sync(fa, a, 16);
  wmma::load_matrix_sync(fb, b, 16);
  wmma::fill_fragment(fc0, 0.0f);
  wmma::fill_fragment(fc1, 0.0f);
  wmma::fill_fragment(fc2, 0.0f);
  wmma::fill_fragment(fc3, 0.0f);
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    wmma::mma_sync(fc0, fa, fb, fc0);
    wmma::mma_sync(fc1, fa, fb, fc1);
    wmma::mma_sync(fc2, fa, fb, fc2);
    wmma::mma_sync(fc3, fa, fb, fc3);
  }
  const int warp = threadIdx.x / 32;
  float* out = c + ((size_t)blockIdx.x * (blockDim.x / 32) + warp) * 1024;
  wmma::store_matrix_sync(out + 0 * 256, fc0, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(out + 1 * 256, fc1, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(out + 2 * 256, fc2, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(out + 3 * 256, fc3, 16, wmma::mem_row_major);
}

__global__ void WmmaBf16Kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                               float* c, int iters) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc0, fc1, fc2, fc3;
  wmma::load_matrix_sync(fa, a, 16);
  wmma::load_matrix_sync(fb, b, 16);
  wmma::fill_fragment(fc0, 0.0f);
  wmma::fill_fragment(fc1, 0.0f);
  wmma::fill_fragment(fc2, 0.0f);
  wmma::fill_fragment(fc3, 0.0f);
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    wmma::mma_sync(fc0, fa, fb, fc0);
    wmma::mma_sync(fc1, fa, fb, fc1);
    wmma::mma_sync(fc2, fa, fb, fc2);
    wmma::mma_sync(fc3, fa, fb, fc3);
  }
  const int warp = threadIdx.x / 32;
  float* out = c + ((size_t)blockIdx.x * (blockDim.x / 32) + warp) * 1024;
  wmma::store_matrix_sync(out + 0 * 256, fc0, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(out + 1 * 256, fc1, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(out + 2 * 256, fc2, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(out + 3 * 256, fc3, 16, wmma::mem_row_major);
}

__device__ __forceinline__ void MmaFp8(float c0, float c1, float c2, float c3,
                                       unsigned int a0, unsigned int a1,
                                       unsigned int a2, unsigned int a3,
                                       unsigned int b0, unsigned int b1,
                                       float& d0, float& d1, float& d2,
                                       float& d3) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
        "f"(c1), "f"(c2), "f"(c3));
}

__global__ void MmaFp8Kernel(unsigned int seed, float* sink, int iters) {
  const unsigned int a0 = seed, a1 = seed + 1, a2 = seed + 2, a3 = seed + 3;
  const unsigned int b0 = seed + 4, b1 = seed + 5;
  float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float c2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float c3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    float d0, d1, d2, d3;
    MmaFp8(c0[0], c0[1], c0[2], c0[3], a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    c0[0] = d0; c0[1] = d1; c0[2] = d2; c0[3] = d3;
    MmaFp8(c1[0], c1[1], c1[2], c1[3], a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    c1[0] = d0; c1[1] = d1; c1[2] = d2; c1[3] = d3;
    MmaFp8(c2[0], c2[1], c2[2], c2[3], a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    c2[0] = d0; c2[1] = d1; c2[2] = d2; c2[3] = d3;
    MmaFp8(c3[0], c3[1], c3[2], c3[3], a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    c3[0] = d0; c3[1] = d1; c3[2] = d2; c3[3] = d3;
  }
  const float r = c0[0] + c0[1] + c0[2] + c0[3] + c1[0] + c1[1] + c1[2] +
                  c1[3] + c2[0] + c2[1] + c2[2] + c2[3] + c3[0] + c3[1] +
                  c3[2] + c3[3];
  if (r == -1.0f) sink[0] = r;
}

__device__ __forceinline__ void MmaFp4Unscaled(float c0, float c1, float c2,
                                               float c3, unsigned int a0,
                                               unsigned int a1, unsigned int a2,
                                               unsigned int a3, unsigned int b0,
                                               unsigned int b1, float& d0,
                                               float& d1, float& d2, float& d3) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
        "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void MmaMxf4(float c0, float c1, float c2, float c3,
                                        unsigned int a0, unsigned int a1,
                                        unsigned int a2, unsigned int a3,
                                        unsigned int b0, unsigned int b1,
                                        unsigned int sfa, unsigned int sfb,
                                        float& d0, float& d1, float& d2,
                                        float& d3) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X"
      ".f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "%14, {0, 0}, %15, {0, 0};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
        "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ void MmaMxf4Nvf4(
    float c0, float c1, float c2, float c3, unsigned int a0, unsigned int a1,
    unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1,
    unsigned int sfa, unsigned int sfb, float& d0, float& d1, float& d2,
    float& d3) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
      ".f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
      "%14, {0, 0}, %15, {0, 0};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
        "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
}

template <int Kind>
__global__ void MmaFp4Kernel(unsigned int seed, float* sink, int iters) {
  const unsigned int a0 = seed, a1 = seed + 1, a2 = seed + 2, a3 = seed + 3;
  const unsigned int b0 = seed + 4, b1 = seed + 5;
  const unsigned int sfa = (Kind == 1) ? 0x7F7F7F7Fu : 0x38383838u;
  const unsigned int sfb = sfa;
  float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float c2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float c3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    float d0, d1, d2, d3;
    if (Kind == 0) {
      MmaFp4Unscaled(c0[0], c0[1], c0[2], c0[3], a0, a1, a2, a3, b0, b1, d0,
                     d1, d2, d3);
    } else if (Kind == 1) {
      MmaMxf4(c0[0], c0[1], c0[2], c0[3], a0, a1, a2, a3, b0, b1, sfa, sfb,
              d0, d1, d2, d3);
    } else {
      MmaMxf4Nvf4(c0[0], c0[1], c0[2], c0[3], a0, a1, a2, a3, b0, b1, sfa,
                  sfb, d0, d1, d2, d3);
    }
    c0[0] = d0; c0[1] = d1; c0[2] = d2; c0[3] = d3;
    if (Kind == 0) {
      MmaFp4Unscaled(c1[0], c1[1], c1[2], c1[3], a0, a1, a2, a3, b0, b1, d0,
                     d1, d2, d3);
    } else if (Kind == 1) {
      MmaMxf4(c1[0], c1[1], c1[2], c1[3], a0, a1, a2, a3, b0, b1, sfa, sfb,
              d0, d1, d2, d3);
    } else {
      MmaMxf4Nvf4(c1[0], c1[1], c1[2], c1[3], a0, a1, a2, a3, b0, b1, sfa,
                  sfb, d0, d1, d2, d3);
    }
    c1[0] = d0; c1[1] = d1; c1[2] = d2; c1[3] = d3;
    if (Kind == 0) {
      MmaFp4Unscaled(c2[0], c2[1], c2[2], c2[3], a0, a1, a2, a3, b0, b1, d0,
                     d1, d2, d3);
    } else if (Kind == 1) {
      MmaMxf4(c2[0], c2[1], c2[2], c2[3], a0, a1, a2, a3, b0, b1, sfa, sfb,
              d0, d1, d2, d3);
    } else {
      MmaMxf4Nvf4(c2[0], c2[1], c2[2], c2[3], a0, a1, a2, a3, b0, b1, sfa,
                  sfb, d0, d1, d2, d3);
    }
    c2[0] = d0; c2[1] = d1; c2[2] = d2; c2[3] = d3;
    if (Kind == 0) {
      MmaFp4Unscaled(c3[0], c3[1], c3[2], c3[3], a0, a1, a2, a3, b0, b1, d0,
                     d1, d2, d3);
    } else if (Kind == 1) {
      MmaMxf4(c3[0], c3[1], c3[2], c3[3], a0, a1, a2, a3, b0, b1, sfa, sfb,
              d0, d1, d2, d3);
    } else {
      MmaMxf4Nvf4(c3[0], c3[1], c3[2], c3[3], a0, a1, a2, a3, b0, b1, sfa,
                  sfb, d0, d1, d2, d3);
    }
    c3[0] = d0; c3[1] = d1; c3[2] = d2; c3[3] = d3;
  }
  const float r = c0[0] + c0[1] + c0[2] + c0[3] + c1[0] + c1[1] + c1[2] +
                  c1[3] + c2[0] + c2[1] + c2[2] + c2[3] + c3[0] + c3[1] +
                  c3[2] + c3[3];
  if (r == -1.0f) sink[0] = r;
}

}  // namespace

int main(int argc, char** argv) {
  const int device = bench::GetIntArg(argc, argv, "--device", 0);
  const int repeat = bench::GetIntArg(argc, argv, "--repeat", 3);
  const std::string out = bench::GetArg(argc, argv, "--out", "");

  CHECK_CUDA(cudaSetDevice(device));
  const bench::DeviceInfo info = bench::GetDeviceInfo(device);
  const int sms = info.prop.multiProcessorCount;
  const int threads = 256;
  const int warps_per_block = threads / 32;
  const int blocks = sms * 4;

  __half* d_h = nullptr;
  __nv_bfloat16* d_b = nullptr;
  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_h, 16 * 16 * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_b, 16 * 16 * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_c, (size_t)blocks * warps_per_block * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_h, 0x3c, 16 * 16 * sizeof(__half)));
  CHECK_CUDA(cudaMemset(d_b, 0x3c, 16 * 16 * sizeof(__nv_bfloat16)));

  std::ofstream csv;
  if (!out.empty()) {
    csv.open(out);
    if (!csv) {
      std::fprintf(stderr, "Cannot open %s\n", out.c_str());
      return EXIT_FAILURE;
    }
    csv << "metric,value,unit,blocks,threads,iters\n";
  }

  auto run = [&](const char* name, auto launcher, int iters,
                 double flops_per_mma) {
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
      bench::GpuTimer timer;
      timer.Start();
      launcher();
      CHECK_CUDA_KERNEL();
      times.push_back(timer.StopMs());
    }
    const double ms = bench::Median(times);
    // Four MMA instructions per warp per iteration.  Each m16n16k16 WMMA or
    // m16n8k32 FP8 MMA performs 16*16*16*2 = 8192 FLOPs.
    const double flops =
        (double)blocks * warps_per_block * iters * 4.0 * flops_per_mma;
    const double tflops = flops / (ms * 1.0e9);
    std::printf("%-16s %10.3f TFLOPS (%.3f ms)\n", name, tflops, ms);
    if (csv) {
      csv << name << "," << std::fixed << std::setprecision(6) << tflops
          << ",TFLOPS," << blocks << "," << threads << "," << iters << "\n";
    }
  };

  // Warm up the tensor-core clock domain.
  for (int w = 0; w < 200; ++w) {
    WmmaFp16Kernel<<<blocks, threads>>>(d_h, d_h, d_c, 1000);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  if (bench::HasArg(argc, argv, "--power")) {
    // Long FP4 kernel for steady-state power/clock sampling.
    MmaFp4Kernel<2><<<blocks, threads>>>(0x11111111, d_c, 80000000);
    CHECK_CUDA_KERNEL();
    CHECK_CUDA(cudaDeviceSynchronize());
    if (csv) csv.close();
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_h));
    return EXIT_SUCCESS;
  }

  run("fp16_wmma", [&] { WmmaFp16Kernel<<<blocks, threads>>>(d_h, d_h, d_c, 20000); }, 20000, 8192.0);
  run("bf16_wmma", [&] { WmmaBf16Kernel<<<blocks, threads>>>(d_b, d_b, d_c, 20000); }, 20000, 8192.0);
  run("fp8_mma", [&] { MmaFp8Kernel<<<blocks, threads>>>(0x3c3c3c3c, d_c, 20000); }, 20000, 8192.0);
  run("fp4_unscaled", [&] { MmaFp4Kernel<0><<<blocks, threads>>>(0x11111111, d_c, 20000); }, 20000, 8192.0);
  run("fp4_mxf4", [&] { MmaFp4Kernel<1><<<blocks, threads>>>(0x11111111, d_c, 20000); }, 20000, 16384.0);
  run("fp4_nvfp4", [&] { MmaFp4Kernel<2><<<blocks, threads>>>(0x11111111, d_c, 20000); }, 20000, 16384.0);

  if (csv) csv.close();
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_h));
  return EXIT_SUCCESS;
}
