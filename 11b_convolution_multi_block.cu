#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

__global__ void convolution1d(const int *a, const int *b, int *out, const size_t n, const size_t convSize) {
    extern __shared__ int shared[];

    // Global index into the full array.
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Local index into the shared array. Shared memory is shared across threads in the same block.
    const size_t localIdx = threadIdx.x;

    const size_t threadsPerBlock = blockDim.x;

    int* shared_data = shared;
    int* shared_conv = shared + threadsPerBlock + convSize;

    // Write the first <threadsPerBlock> data elements to shared memory.
    if (idx < n) {
        shared_data[localIdx] = a[idx];
    }
    // Write the full convolution kernel to shared memory.
    if (localIdx < convSize) {
        shared_conv[localIdx] = b[localIdx];
    }
    // We want to use the threads that are unused for copying the convolution kernel to shared memory
    // to copy the missing data from the input array to shared memory.
    else if (localIdx < 2 * convSize) {
        shared_data[localIdx - convSize + threadsPerBlock] = a[idx - convSize + threadsPerBlock];
    }

    __syncthreads();

    if (idx < n) {
        int sum = 0;
        for (size_t k = 0; k < convSize; k++) {
            sum += shared_data[localIdx + k] * shared_conv[k];
        }
        out[idx] = sum;
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
    const size_t n = 15;
    const size_t convSize = 4;
    int hostInA[n];
    int hostInB[convSize];
    int hostOut[n];
    for (size_t i = 0; i < n; i++) {
        hostInA[i] = i;
        std::cout << hostInA[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < convSize; i++) {
        hostInB[i] = i;
        std::cout << hostInB[i] << " ";
    }
    std::cout << std::endl;

    // Allocate memory on the device for both input and output arrays, which have the same size.
    int *deviceInA, *deviceInB, *deviceOut;
    CUDA_CHECK(cudaMalloc((void **)&deviceInA, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&deviceInB, convSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&deviceOut, n *sizeof(int)));

    // Copy the input array from the host to the device.
    CUDA_CHECK(cudaMemcpy(deviceInA, hostInA, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceInB, hostInB, convSize * sizeof(int), cudaMemcpyHostToDevice));

    // Define the number of threads per block and the number of blocks.
    const int threadsPerBlock = 8;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Make sure there are enough threads to copy the convolution kernel to shared memory
    // and to copy the overlapping data from the input array to shared memory.
    assert(threadsPerBlock <= convSize * 2);

    std::cout << "blocks: " << blocks << ", threadsPerBlock: " << threadsPerBlock << std::endl;

    const int sharedMemSize = (threadsPerBlock + 2 * convSize) * sizeof(int);
    convolution1d<<<blocks, threadsPerBlock, sharedMemSize>>>(deviceInA, deviceInB, deviceOut, n, convSize);
    CUDA_CHECK(cudaGetLastError()); // Check for any errors in the kernel launch.
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all threads to complete.

    // Copy the output array from the device to the host.
    CUDA_CHECK(cudaMemcpy(hostOut, deviceOut, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the output array.
    for (size_t i = 0; i < n; i++) {
        std::cout << hostOut[i] << " ";
    }
    std::cout << std::endl;

    // Release the allocated memory on the device.
    CUDA_CHECK(cudaFree(deviceInA));
    CUDA_CHECK(cudaFree(deviceInB));
    CUDA_CHECK(cudaFree(deviceOut));

    return 0;
}
