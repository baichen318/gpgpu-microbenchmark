// On-chip network (NoC) latency, topology, and bandwidth scaling.
#include "common.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace {

__global__ void NoCPingPongKernel(int* seq, int* sender, int* ack,
                                  unsigned long long* lat, int* smids,
                                  int nblocks, int repeats) {
  cg::grid_group grid = cg::this_grid();
  const int bid = blockIdx.x;
  if (threadIdx.x == 0) smids[bid] = (int)bench::SmId();
  grid.sync();

  for (int rep = 0; rep < repeats; ++rep) {
    for (int r = 0; r < nblocks; ++r) {
      const int target = (bid + r) % nblocks;
      const int tag = rep * nblocks + r + 1;
      if (threadIdx.x == 0 && target != bid) {
        const unsigned long long start = bench::GlobalTimer();
        sender[target] = bid;
        __threadfence();
        atomicExch(&seq[target], tag);

        while (atomicAdd(&seq[bid], 0) != tag) {
          __nanosleep(8);
        }
        __threadfence();
        const int from = sender[bid];
        atomicExch(&ack[from], tag);

        while (atomicAdd(&ack[bid], 0) != tag) {
          __nanosleep(8);
        }
        const unsigned long long end = bench::GlobalTimer();
        lat[(size_t)bid * nblocks + target] += (end - start);
      }
      grid.sync();
    }
  }
}

__global__ void NoCBandwidthKernel(const float4* __restrict__ data, size_t n4,
                                   int iters, float4* __restrict__ sink) {
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int it = 0; it < iters; ++it) {
    for (size_t i = threadIdx.x; i < n4; i += blockDim.x) {
      const float4 v = bench::LoadGlobalFloat4(&data[i]);
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
  }
  if (acc.x == -1.0f) *sink = acc;
}

int MaxCooperativeBlocks(int threads, size_t shmem) {
  int blocks = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, NoCPingPongKernel, threads, shmem);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return blocks;
}

}  // namespace

int main(int argc, char** argv) {
  const int device = bench::GetIntArg(argc, argv, "--device", 0);
  const int repeat = bench::GetIntArg(argc, argv, "--repeat", 3);
  const std::string out_prefix = bench::GetArg(argc, argv, "--out-prefix", "");

  CHECK_CUDA(cudaSetDevice(device));
  const bench::DeviceInfo info = bench::GetDeviceInfo(device);
  const int sms = info.prop.multiProcessorCount;
  const int threads = 128;
  const size_t shmem = 96 * 1024;
  std::printf("GPU: %s, SMs=%d, cooperative_launch=%d, cluster_launch=%d\n",
              info.prop.name, sms, info.cooperative_launch, info.cluster_launch);

  if (!info.cooperative_launch) {
    std::printf("Cooperative launch is not supported; skipping NoC tests.\n");
    return EXIT_SUCCESS;
  }

  CHECK_CUDA(cudaFuncSetAttribute(NoCPingPongKernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)shmem));
  const int max_blocks = MaxCooperativeBlocks(threads, shmem);
  std::printf("Cooperative occupancy at %zu bytes shared: %d block(s)/SM\n",
              shmem, max_blocks);
  if (max_blocks < 1) {
    std::printf("Cannot fit a cooperative block with the requested shared "
                "memory; skipping NoC latency test.\n");
    return EXIT_SUCCESS;
  }

  int* d_seq = nullptr;
  int* d_sender = nullptr;
  int* d_ack = nullptr;
  int* d_smids = nullptr;
  unsigned long long* d_lat = nullptr;
  CHECK_CUDA(cudaMalloc(&d_seq, sms * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_sender, sms * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_ack, sms * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_smids, sms * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_lat, (size_t)sms * sms * sizeof(unsigned long long)));

  // -----------------------------------------------------------------------
  // NoC bandwidth scaling: one block per SM up to all SMs, each block reads
  // the same L2-resident 16 MiB buffer.
  // -----------------------------------------------------------------------
  const size_t bw_bytes = 16ull << 20;
  float4* d_bw = nullptr;
  float4* d_sink = nullptr;
  CHECK_CUDA(cudaMalloc(&d_bw, bw_bytes));
  CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float4)));
  CHECK_CUDA(cudaMemset(d_bw, 7, bw_bytes));
  const size_t n4 = bw_bytes / sizeof(float4);
  const int bw_iters = 20;

  std::ofstream bw_csv;
  if (!out_prefix.empty()) {
    bw_csv.open(out_prefix + "_bandwidth.csv");
    if (bw_csv) bw_csv << "blocks,threads,iters,time_ms,bandwidth_gbps\n";
  }
  std::printf("\nNoC/L2 bandwidth scaling:\n");
  std::printf("%8s %12s %12s\n", "blocks", "GB/s", "GB/s/SM");
  const int block_counts[] = {1, 2, 4, 8, 16, 32, 64, 94, 128, 188};
  for (int blocks : block_counts) {
    if (blocks > sms) continue;
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
      bench::GpuTimer timer;
      timer.Start();
      NoCBandwidthKernel<<<blocks, 1024>>>(d_bw, n4, bw_iters, d_sink);
      CHECK_CUDA_KERNEL();
      times.push_back(timer.StopMs());
    }
    const double ms = bench::Median(times);
    const size_t bytes = bw_bytes * (size_t)bw_iters * (size_t)blocks;
    const double gbps = bench::MsToGBps(bytes, ms);
    std::printf("%8d %12.1f %12.1f\n", blocks, gbps, gbps / blocks);
    if (bw_csv) {
      bw_csv << blocks << ",1024," << bw_iters << "," << std::fixed
             << std::setprecision(6) << ms << "," << gbps << "\n";
    }
  }
  if (bw_csv) bw_csv.close();

  // -----------------------------------------------------------------------
  // NoC latency / topology: all-to-all ping-pong, rotating target.
  // -----------------------------------------------------------------------
  std::ofstream lat_csv;
  std::ofstream matrix_csv;
  std::ofstream summary_csv;
  if (!out_prefix.empty()) {
    lat_csv.open(out_prefix + "_latency.csv");
    if (lat_csv) {
      lat_csv << "active_blocks,pairs,median_roundtrip_ns,median_oneway_ns,"
                 "min_roundtrip_ns,max_roundtrip_ns\n";
    }
    summary_csv.open(out_prefix + "_summary.csv");
    if (summary_csv) {
      summary_csv << "metric,value,unit\n";
    }
  }

  std::printf("\nNoC ping-pong latency vs active SMs:\n");
  std::printf("%14s %12s %12s %12s\n", "active_blocks", "roundtrip_ns",
              "oneway_ns", "pairs");
  const int active_counts[] = {2, 4, 8, 16, 32, 64, 94, 128, 188};
  const int repeats = 5;
  for (int active : active_counts) {
    if (active > sms) continue;
    const size_t lat_elems = (size_t)active * active;
    CHECK_CUDA(cudaMemsetAsync(d_seq, 0, active * sizeof(int)));
    CHECK_CUDA(cudaMemsetAsync(d_sender, 0, active * sizeof(int)));
    CHECK_CUDA(cudaMemsetAsync(d_ack, 0, active * sizeof(int)));
    CHECK_CUDA(cudaMemsetAsync(d_lat, 0, lat_elems * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemsetAsync(d_smids, 0, active * sizeof(int)));

    void* args[] = {&d_seq, &d_sender, &d_ack, &d_lat, &d_smids,
                    (void*)&active, (void*)&repeats};
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)NoCPingPongKernel, dim3(active), dim3(threads), args, shmem, 0);
    if (err != cudaSuccess) {
      std::printf("  active=%d: cooperative launch failed: %s\n", active,
                  cudaGetErrorString(err));
      cudaGetLastError();
      continue;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<unsigned long long> lat(lat_elems, 0);
    std::vector<int> smids(active, 0);
    CHECK_CUDA(cudaMemcpy(lat.data(), d_lat,
                          lat_elems * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(smids.data(), d_smids, active * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<double> pair_ns;
    pair_ns.reserve(lat_elems);
    for (int i = 0; i < active; ++i) {
      for (int j = 0; j < active; ++j) {
        if (i == j) continue;
        pair_ns.push_back((double)lat[(size_t)i * active + j] / repeats);
      }
    }
    const double med = bench::Median(pair_ns);
    const double mn = bench::Min(pair_ns);
    const double mx = bench::Max(pair_ns);
    std::printf("%14d %12.1f %12.1f %12zu\n", active, med, med * 0.5,
                pair_ns.size());
    if (lat_csv) {
      lat_csv << active << "," << pair_ns.size() << "," << std::fixed
              << std::setprecision(3) << med << "," << med * 0.5 << "," << mn
              << "," << mx << "\n";
    }

    if (active == sms) {
      if (!out_prefix.empty()) {
        matrix_csv.open(out_prefix + "_matrix.csv");
        if (matrix_csv) {
          matrix_csv << "block_i,block_j,smid_i,smid_j,roundtrip_ns,oneway_ns\n";
        }
      }
      for (int i = 0; i < active; ++i) {
        for (int j = 0; j < active; ++j) {
          if (i == j) continue;
          const double rt = (double)lat[(size_t)i * active + j] / repeats;
          if (matrix_csv) {
            matrix_csv << i << "," << j << "," << smids[i] << "," << smids[j]
                       << "," << std::fixed << std::setprecision(3) << rt << ","
                       << rt * 0.5 << "\n";
          }
        }
      }
      if (matrix_csv) matrix_csv.close();

      if (summary_csv) {
        summary_csv << "active_blocks," << active << ",blocks\n";
        summary_csv << "pairs," << pair_ns.size() << ",pairs\n";
        summary_csv << "median_roundtrip_ns," << std::fixed
                    << std::setprecision(3) << med << ",ns\n";
        summary_csv << "median_oneway_ns," << med * 0.5 << ",ns\n";
        summary_csv << "min_roundtrip_ns," << mn << ",ns\n";
        summary_csv << "max_roundtrip_ns," << mx << ",ns\n";
        summary_csv << "mean_roundtrip_ns," << bench::Mean(pair_ns) << ",ns\n";
        summary_csv << "stddev_roundtrip_ns," << bench::Stddev(pair_ns) << ",ns\n";
      }
    }
  }
  if (lat_csv) lat_csv.close();
  if (summary_csv) summary_csv.close();

  CHECK_CUDA(cudaFree(d_lat));
  CHECK_CUDA(cudaFree(d_smids));
  CHECK_CUDA(cudaFree(d_ack));
  CHECK_CUDA(cudaFree(d_sender));
  CHECK_CUDA(cudaFree(d_seq));
  CHECK_CUDA(cudaFree(d_sink));
  CHECK_CUDA(cudaFree(d_bw));
  return EXIT_SUCCESS;
}
