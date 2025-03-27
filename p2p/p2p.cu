#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
	if (err != cudaSuccess) { \
	    fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
		    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
		    exit(EXIT_FAILURE); \
	} \
}

int main(int argc, char *argv[]) {
	// Hardware initialization.
	int num_devices;
	CHECK_CUDA(cudaGetDeviceCount(&num_devices));
	if (num_devices < 2) {
	    fprintf(stderr, "We need at least 2 GPUs.\n");
		return -1;
	}

	// Select GPU0 & GPU1.
	int src_gpu = 0, dest_gpu = 1;
	CHECK_CUDA(cudaSetDevice(src_gpu));
	cudaDeviceEnablePeerAccess(1, 0);
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, src_gpu));
	printf("GPU information:\nsource GPU %d: %s (PCIe %d.%d)\n",
		src_gpu, prop.name, prop.pciBusID, prop.pciDeviceID);

	// Check NVLink information.
	// printf("\nNVLink information:\n");
	// system("nvidia-smi nvlink --status");
	// system("nvidia-smi topo -m");

	// Memory allocation.
	const size_t data_size = 1 << 28; // 256 MB
	int *h_src, *h_dest;
	int *d_src, *d_dest;

	// Allocate page-locked host memory to improve the performance(@ref).
	CHECK_CUDA(cudaMallocHost((void**)&h_src, data_size));
	CHECK_CUDA(cudaMallocHost((void**)&h_dest, data_size));
	CHECK_CUDA(cudaMalloc(&d_src, data_size));
	CHECK_CUDA(cudaSetDevice(dest_gpu));
	CHECK_CUDA(cudaMalloc(&d_dest, data_size));

	// Initialize the data.
	for (size_t i = 0; i < data_size / sizeof(int); ++i) {
	    h_src[i] = i % 256;
	}

	// Create CUDA events.
	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	// Execute the microbenchmark.
	const int iterations = 1000;
	float total_time = 0.0f;

	for (int i = 0; i < iterations; ++i) {
		// Host -> source device.
		CHECK_CUDA(cudaMemcpy(d_src, h_src, data_size, cudaMemcpyHostToDevice));

		// Source device -> target device.
		CHECK_CUDA(cudaEventRecord(start)); 
		CHECK_CUDA(cudaMemcpyPeer(d_dest, dest_gpu, d_src, src_gpu, data_size));
		CHECK_CUDA(cudaEventSynchronize(stop));
		CHECK_CUDA(cudaEventRecord(stop));

		// Target device -> host.
		CHECK_CUDA(cudaMemcpy(h_dest, d_dest, data_size, cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaStreamSynchronize(0));

		float elapsed_time;
		CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
		total_time += elapsed_time;
	}

	// Calculate the transmission time & bandwidth.
	double avg_time = total_time / iterations;
	double bandwidth = (data_size / (1e6)) / avg_time; // MB/ms = GB/s
	printf("\nAverage transmission time: %.8f s, average single direction bandwidth: %.8f GB/s\n",
		avg_time, bandwidth);

	// Free resources.
	CHECK_CUDA(cudaFreeHost(h_src));
	CHECK_CUDA(cudaFreeHost(h_dest));
	CHECK_CUDA(cudaFree(d_src));
	CHECK_CUDA(cudaFree(d_dest));
	CHECK_CUDA(cudaEventDestroy(start));
	CHECK_CUDA(cudaEventDestroy(stop));

	return 0;
}
