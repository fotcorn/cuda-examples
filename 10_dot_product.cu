#include <cuda_runtime.h>
#include <iostream>

__global__ void dot_product(const int *a, const int *b, int *out, const size_t n) {
    extern __shared__ int shared[];

    // Global index into the full array.
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Local index into the shared array. Shared memory is shared across threads in the same block.
    const size_t localIdx = threadIdx.x;

    if (idx < n) {
        shared[localIdx] = a[idx] * b[idx];
        __syncthreads();
    }

    if (idx == 0) {
        int sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += shared[i];
        }
        out[0] = sum;
    }
}


#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)


int main() {
    // Check if there is a GPU available.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Initialize the input array on the host.
    const size_t n = 8;
    int hostInA[n];
    int hostInB[n];
    int hostOut[1];
    for (size_t i = 0; i < n; i++) {
        hostInA[i] = i;
        hostInB[i] = i;
    }

    // Allocate memory on the device for both input and output arrays, which have the same size.
    int *deviceInA, *deviceInB, *deviceOut;
    CUDA_CHECK(cudaMalloc((void **)&deviceInA, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&deviceInB, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&deviceOut, sizeof(int)));

    // Copy the input array from the host to the device.
    CUDA_CHECK(cudaMemcpy(deviceInA, hostInA, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceInB, hostInB, n * sizeof(int), cudaMemcpyHostToDevice));

    // Define the number of threads per block and the number of blocks.
    const int threadsPerBlock = 256; // Standard value, good balance between resource usage (registers and shared memory) and performance.
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "blocks: " << blocks << ", threadsPerBlock: " << threadsPerBlock << std::endl;
    dot_product<<<blocks, threadsPerBlock, n * sizeof(int)>>>(deviceInA, deviceInB, deviceOut, n);
    CUDA_CHECK(cudaGetLastError()); // Check for any errors in the kernel launch.
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all threads to complete.

    // Copy the output array from the device to the host.
    CUDA_CHECK(cudaMemcpy(hostOut, deviceOut, sizeof(int), cudaMemcpyDeviceToHost));

    // Print the output array.
    std::cout << hostOut[0] << std::endl;

    // Release the allocated memory on the device.
    CUDA_CHECK(cudaFree(deviceInA));
    CUDA_CHECK(cudaFree(deviceInB));
    CUDA_CHECK(cudaFree(deviceOut));

    return 0;
}

