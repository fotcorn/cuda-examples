#include <cuda_runtime.h>
#include <iostream>

__global__ void prefix_sum(const int *in, int *out, const size_t n) {
    extern __shared__ int shared[];

    // Global index into the full array.
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Local index into the shared array. Shared memory is shared across threads in the same block.
    const size_t localIdx = threadIdx.x;

    int numContractions = ceilf(log2f(blockDim.x));

    if (idx < n) {
        shared[localIdx] = in[idx];
    } else {
        shared[localIdx] = 0;
    }

    __syncthreads();

    for (int c = 0; c < numContractions; c++) {
        shared[localIdx] = shared[localIdx] + shared[localIdx + (1 << c)];
    }

    __syncthreads();

    if (idx % blockDim.x == 0) {
        out[idx / blockDim.x] = shared[localIdx];
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

    const size_t n = 15;

    // Define the number of threads per block and the number of blocks.
    const int threadsPerBlock = 8;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize the input array on the host.
    int hostIn[n];
    int hostOut[blocks];
    for (size_t i = 0; i < n; i++) {
        hostIn[i] = i;
    }

    // Allocate memory on the device for both input and output arrays, which have the same size.
    int *deviceIn, *deviceOut;
    CUDA_CHECK(cudaMalloc((void **)&deviceIn, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&deviceOut, n * sizeof(int)));

    // Copy the input array from the host to the device.
    CUDA_CHECK(cudaMemcpy(deviceIn, hostIn, n * sizeof(int), cudaMemcpyHostToDevice));

    std::cout << "blocks: " << blocks << ", threadsPerBlock: " << threadsPerBlock << std::endl;
    prefix_sum<<<blocks, threadsPerBlock, n * sizeof(int)>>>(deviceIn, deviceOut, n);

    CUDA_CHECK(cudaGetLastError()); // Check for any errors in the kernel launch.
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all threads to complete.

    // Copy the output array from the device to the host.
    CUDA_CHECK(cudaMemcpy(hostOut, deviceOut, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the output array.
    for (size_t i = 0; i < blocks; i++) {
        std::cout << hostOut[i] << " ";
    }
    std::cout << std::endl;

    // Release the allocated memory on the device.
    CUDA_CHECK(cudaFree(deviceIn));
    CUDA_CHECK(cudaFree(deviceOut));

    return 0;
}
