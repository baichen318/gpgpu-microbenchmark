// Author: eecbai@ust.hk

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>


#define SM_NUM_START 1
#define SM_NUM_END 1024


#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void _peer2peer_store(int *d_src, int *d_dest, size_t num_elements) {
    size_t idx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    size_t thread_per_block = blockDim.x * blockDim.y * blockDim.z;
    size_t blk_idx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    // int value;
    for (int i = idx; i < num_elements; i += thread_per_block){
        // __asm__ volatile (
        //     "ld.volatile.global.u32 %0, [%1];"
        //     :
        //     : "r"(value), "l"(&d_src[blk_idx*num_elements+i])
        // );

        // __asm__ volatile (
        //     "st.volatile.global.u32 [%0], %1;"
        //     :
        //     : "l"(&d_dest[blk_idx*num_elements+i]), "r"(value)
        // );

        // __asm__ volatile (
        //     "ld.global.u32 %0, [%1];"
        //     :
        //     : "r"(value), "l"(&d_src[blk_idx*num_elements+i])
        // );

        // __asm__ volatile (
        //     "st.global.u32 [%0], %1;"
        //     :
        //     : "l"(&d_dest[blk_idx*num_elements+i]), "r"(value)
        // );
        // d_dest[blk_idx * num_elements + i] = d_src[blk_idx * num_elements + i];
        d_dest[blk_idx * num_elements + i] = d_src[blk_idx * num_elements + i];
    }
}

__global__ void _peer2peer_load(int *d_src, int *d_dest, size_t num_elements) {
    size_t idx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    size_t thread_per_block = blockDim.x * blockDim.y * blockDim.z;
    size_t blk_idx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    // int value;
    for (int i = idx; i < num_elements; i += thread_per_block){
        // __asm__ volatile (
        //     "ld.volatile.global.u32 %0, [%1];"
        //     :
        //     : "r"(value), "l"(&d_src[blk_idx*num_elements+i])
        // );

        // __asm__ volatile (
        //     "st.volatile.global.u32 [%0], %1;"
        //     :
        //     : "l"(&d_dest[blk_idx*num_elements+i]), "r"(value)
        // );

        // __asm__ volatile (
        //     "ld.global.u32 %0, [%1];"
        //     :
        //     : "r"(value), "l"(&d_src[blk_idx*num_elements+i])
        // );

        // __asm__ volatile (
        //     "st.global.u32 [%0], %1;"
        //     :
        //     : "l"(&d_dest[blk_idx*num_elements+i]), "r"(value)
        // );
        // d_dest[blk_idx * num_elements + i] = d_src[blk_idx * num_elements + i];
        d_src[blk_idx * num_elements + i] = d_dest[blk_idx * num_elements + i];
    }
}

void peer2peer_store(int src_gpu, int dest_gpu) {
    // Memory allocation.
    const size_t data_size = 1 << 28; // 256 MB
    size_t num_elements = data_size / sizeof(int);
    int *h_src, *h_dest;
    int *d_src, *d_dest;

    // Allocate page-locked host memory to improve the performance(@ref).
    CHECK_CUDA(cudaMallocHost((void**)&h_src, data_size));
    CHECK_CUDA(cudaMallocHost((void**)&h_dest, data_size));
    // Initialize the data.
    for (size_t i = 0; i < data_size / sizeof(int); ++i) {
        h_src[i] = i % 256;
        h_dest[i] = i % 256;
    }

    // Host -> device.
    CHECK_CUDA(cudaSetDevice(src_gpu));
    CHECK_CUDA(cudaMalloc(&d_src, data_size));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaSetDevice(dest_gpu));
    CHECK_CUDA(cudaMalloc(&d_dest, data_size));
    CHECK_CUDA(cudaMemcpy(d_dest, h_dest, data_size, cudaMemcpyHostToDevice));

    // Create CUDA stream & events.
    CHECK_CUDA(cudaSetDevice(src_gpu));
    cudaStream_t stream;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Execute the microbenchmark.
    const int iterations = 100;

    printf("Peer-to-peer STORE experiments:\n");
    for(size_t sm_num = SM_NUM_START; sm_num <= SM_NUM_END; sm_num *= 2) {
        float total_time = 0.0f;
        dim3 blockDim(1024);
        dim3 gridDim(sm_num);
        size_t num_elements_per_block = num_elements / sm_num;
        // Let driver determines the best allocation strategy for L1 $ size & shared memory size.
        CHECK_CUDA(cudaFuncSetCacheConfig(_peer2peer_store, cudaFuncCachePreferNone));
        printf("sm_num: %d, num_elements: %d, num_elements_per_block: %d, num_elements_per_thread: %f\n",
            sm_num, num_elements, num_elements_per_block, num_elements_per_block / 1024.0);
        for (int i = 0; i < iterations; ++i) {
            CHECK_CUDA(cudaEventRecord(start));
            _peer2peer_store<<<gridDim, blockDim, 0, stream>>>(d_src, d_dest, num_elements_per_block);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            cudaEventSynchronize(0);
            float elapsed_time;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
            total_time += elapsed_time;
        }

        // Calculate the transmission time & bandwidth.
        double avg_time = total_time / iterations;
        double bandwidth = (data_size / (1e6)) / avg_time;; //  MB/ms = GB/s
        // The peer-to-peer NVLink bandwidth for both H100 & H20 is 900 GB/s.
        // So, a single direction achieves 450 GB/s.
        double efficiency = bandwidth / 450 * 100;
        printf("SM_STORE--SM_NUM: %zu, Average transmission time: %.8f ms, \
            average single direction bandwidth: %.8f GB/s, efficiency: %.4f %%\n",
            sm_num, avg_time, bandwidth, efficiency);
    }
    // Free resources.
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dest));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

void peer2peer_load(int src_gpu, int dest_gpu) {
    // Memory allocation.
    const size_t data_size = 1 << 28; // 256 MB
    size_t num_elements = data_size / sizeof(int);
    int *h_src, *h_dest;
    int *d_src, *d_dest;

    // Allocate page-locked host memory to improve the performance(@ref).
    CHECK_CUDA(cudaMallocHost((void**)&h_src, data_size));
    CHECK_CUDA(cudaMallocHost((void**)&h_dest, data_size));
    // Initialize the data.
    for (size_t i = 0; i < data_size / sizeof(int); ++i) {
        h_src[i] = i % 256;
        h_dest[i] = i % 256;
    }

    // Host -> device.
    CHECK_CUDA(cudaSetDevice(src_gpu));
    CHECK_CUDA(cudaMalloc(&d_src, data_size));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaSetDevice(dest_gpu));
    CHECK_CUDA(cudaMalloc(&d_dest, data_size));
    CHECK_CUDA(cudaMemcpy(d_dest, h_dest, data_size, cudaMemcpyHostToDevice));

    // Create CUDA stream & events.
    CHECK_CUDA(cudaSetDevice(src_gpu));
    cudaStream_t stream;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Execute the microbenchmark.
    const int iterations = 100;

    printf("Peer-to-peer LOAD experiments:\n");
    for(size_t sm_num = SM_NUM_START; sm_num <= SM_NUM_END; sm_num *= 2) {
        float total_time = 0.0f;
        dim3 blockDim(1024);
        dim3 gridDim(sm_num);
        size_t num_elements_per_block = num_elements / sm_num;
        // Let driver determines the best allocation strategy for L1 $ size & shared memory size.
        CHECK_CUDA(cudaFuncSetCacheConfig(_peer2peer_load, cudaFuncCachePreferNone));
        printf("sm_num: %d, num_elements: %d, num_elements_per_block: %d, num_elements_per_thread: %f\n",
            sm_num, num_elements, num_elements_per_block, num_elements_per_block / 1024.0);
        for (int i = 0; i < iterations; ++i) {
            CHECK_CUDA(cudaEventRecord(start));
            _peer2peer_load<<<gridDim, blockDim, 0, stream>>>(d_src, d_dest, num_elements_per_block);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            cudaEventSynchronize(0);
            float elapsed_time;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
            total_time += elapsed_time;
        }

        // Calculate the transmission time & bandwidth.
        double avg_time = total_time / iterations;
        double bandwidth = (data_size / (1e6)) / avg_time;; //  MB/ms = GB/s
        // The peer-to-peer NVLink bandwidth for both H100 & H20 is 900 GB/s.
        // So, a single direction achieves 450 GB/s.
        double efficiency = bandwidth / 450 * 100;
        printf("SM_STORE--SM_NUM: %zu, Average transmission time: %.8f ms, \
            average single direction bandwidth: %.8f GB/s, efficiency: %.4f %%\n",
            sm_num, avg_time, bandwidth, efficiency);
    }
    // Free resources.
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dest));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
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
    CHECK_CUDA(cudaDeviceEnablePeerAccess(dest_gpu, 0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, src_gpu));
    printf("GPU information:\nsource GPU %d: %s (PCIe %d.%d)\n",
           src_gpu, prop.name, prop.pciBusID, prop.pciDeviceID);
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dest_gpu));
    printf("destination GPU %d: %s (PCIe %d.%d)\n",
           dest_gpu, prop.name, prop.pciBusID, prop.pciDeviceID);

    peer2peer_store(src_gpu, dest_gpu);
    printf("\n");
    peer2peer_load(src_gpu, dest_gpu);

    return 0;
}
