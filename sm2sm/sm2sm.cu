#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void sm_connectivity_test(int *buffer, int *sm_log, int stride) {
    // Get current SM ID (requires PTX instruction)
    unsigned int sm_id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(sm_id));
    
    // Calculate target SM and data address
    const int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Log SM data usage
    atomicAdd(&buffer[data_index], 1);  // Track data access
    atomicAdd(&sm_log[sm_id], 1);        // Record SM activity
    
    // Cross-SM data interaction
    atomicAdd(&buffer[(data_index + stride) % (gridDim.x * blockDim.x)], 
             buffer[data_index]);
}

int main(int argc, char *argv[]) {
	if(argc < 2) {
        printf("Usage: %s <stride>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

	const int stride = atoi(argv[1]);

    const int total_sms = 78;
    const int block_size = 256;
    const int data_size = total_sms * block_size;
    
    // Allocate tagged buffers
    int *d_buffer, *d_sm_log;
    CHECK_CUDA(cudaMalloc(&d_buffer, data_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sm_log, total_sms * sizeof(int)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemset(d_buffer, 0, data_size * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_sm_log, 0, total_sms * sizeof(int)));

    // Execute enhanced test
    dim3 blocks(total_sms, 1);
    dim3 threads(block_size, 1);
    sm_connectivity_test<<<blocks, threads>>>(d_buffer, d_sm_log, stride);

    // Retrieve results
    int *h_sm_log = new int[total_sms];
    CHECK_CUDA(cudaMemcpy(h_sm_log, d_sm_log, total_sms * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Print SM activity log
    printf("SM Activity Report:\n");
    for(int i=0; i<total_sms; i++) {
        printf("SM %2d: %6d ops | Data Range: %4d-%4d\n", 
              i, h_sm_log[i], 
              i*block_size, (i+1)*block_size-1);
    }

    // Cleanup resources
    delete[] h_sm_log;
    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaFree(d_sm_log));
    return 0;
}