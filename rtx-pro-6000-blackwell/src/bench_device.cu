// Device and topology report for the RTX PRO 6000 Blackwell.
#include "common.cuh"

namespace {

__global__ void DummyKernel() {}

void OccupancyTable(const bench::DeviceInfo& info, std::ostream& os) {
  os << "block_size,max_blocks_per_sm,max_warps_per_sm,occupancy_percent\n";
  const int sms = info.prop.multiProcessorCount;
  const int max_threads = info.prop.maxThreadsPerMultiProcessor;
  const int max_blocks = info.prop.maxBlocksPerMultiProcessor;
  const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  for (int bs : block_sizes) {
    int blocks = 0;
    cudaError_t err =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, DummyKernel, bs, 0);
    if (err != cudaSuccess) {
      os << bs << ",error,error,error\n";
      cudaGetLastError();
      continue;
    }
    if (blocks > max_blocks) blocks = max_blocks;
    const double warps = (double)blocks * bs / info.prop.warpSize;
    const double occupancy = 100.0 * (double)blocks * bs / (double)max_threads;
    os << bs << "," << blocks << "," << warps << "," << std::fixed
       << std::setprecision(2) << occupancy << "\n";
  }
  (void)sms;
}

}  // namespace

int main(int argc, char** argv) {
  int device = bench::GetIntArg(argc, argv, "--device", 0);
  CHECK_CUDA(cudaSetDevice(device));
  bench::DeviceInfo info = bench::GetDeviceInfo(device);
  info.max_cluster_size = bench::MaxPotentialClusterSize((const void*)DummyKernel, 128);
  info.max_blocks_per_cluster = info.max_cluster_size;

  bench::PrintHeader("RTX PRO 6000 Blackwell - device information");
  bench::PrintDeviceInfo(info);

  std::printf("\nOccupancy limits (empty kernel):\n");
  std::printf("  block_size  max_blocks/SM  max_warps/SM  occupancy\n");
  const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  for (int bs : block_sizes) {
    int blocks = 0;
    cudaError_t err =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, DummyKernel, bs, 0);
    if (err != cudaSuccess) {
      std::printf("  %10d  error\n", bs);
      cudaGetLastError();
      continue;
    }
    if (blocks > info.prop.maxBlocksPerMultiProcessor)
      blocks = info.prop.maxBlocksPerMultiProcessor;
    const double warps = (double)blocks * bs / info.prop.warpSize;
    const double occupancy =
        100.0 * (double)blocks * bs / (double)info.prop.maxThreadsPerMultiProcessor;
    std::printf("  %10d  %12d  %12.0f  %8.2f%%\n", bs, blocks, warps, occupancy);
  }

  const std::string out = bench::GetArg(argc, argv, "--out", "");
  if (!out.empty()) {
    std::ofstream csv(out);
    if (!csv) {
      std::fprintf(stderr, "Cannot open %s\n", out.c_str());
      return EXIT_FAILURE;
    }
    bench::WriteDeviceInfoCsv(info, csv);
    csv.close();

    std::string occ_path = out;
    const size_t slash = occ_path.find_last_of('/');
    const size_t dot = occ_path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
      occ_path = occ_path.substr(0, dot) + "_occupancy.csv";
    } else {
      occ_path += "_occupancy.csv";
    }
    std::ofstream occ(occ_path);
    if (occ) OccupancyTable(info, occ);
  }

  return EXIT_SUCCESS;
}
