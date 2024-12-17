#include <cuda_runtime.h>
#include <iostream>

__global__ void map(const int *inA, const int *inB, int *out, const size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = idy * n + idx;
    if (idx < n && idy < n) {
        out[index] = inA[idx] + inB[idy];
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
    const size_t n = 100;
    int hostInX[n];
    int hostInY[n];
    int hostOut[n][n];
    for (size_t i = 0; i < n; i++) {
        hostInX[i] = i;
        hostInY[i] = i * 2;
    }

    // Allocate memory on the device for both input and output arrays, which have the same size.
    int *deviceInX, *deviceInY, *deviceOut;
    CUDA_CHECK(cudaMalloc((void **)&deviceInX, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&deviceInY, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&deviceOut, n * n * sizeof(int)));

    // Copy the input array from the host to the device.
    CUDA_CHECK(cudaMemcpy(deviceInX, hostInX, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceInY, hostInY, n * sizeof(int), cudaMemcpyHostToDevice));

    // Define the number of threads per block and the number of blocks.
    const int threads = 32;
    dim3 threadsPerBlock(threads, threads);
    const int blocks = (n + threads - 1) / threads;
    dim3 blocksPerGrid(blocks, blocks);

    std::cout << "blocks: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << "), threadsPerBlock: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;
    map<<<blocksPerGrid, threadsPerBlock>>>(deviceInX, deviceInY, deviceOut, n);
    CUDA_CHECK(cudaGetLastError()); // Check for any errors in the kernel launch.
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all threads to complete.

    // Copy the output array from the device to the host.
    CUDA_CHECK(cudaMemcpy(hostOut, deviceOut, n * n *sizeof(int), cudaMemcpyDeviceToHost));

    // Print the output array.
    for (size_t y = 0; y < n; y++) {
        std::cout << "# " << y << std::endl;
        for (size_t x = 0; x < n; x++) {
            std::cout << hostOut[y][x] << ' ';
        }
        std::cout << std::endl;
    }

    // Release the allocated memory on the device.
    CUDA_CHECK(cudaFree(deviceInX));
    CUDA_CHECK(cudaFree(deviceInY));
    CUDA_CHECK(cudaFree(deviceOut));

    return 0;
}
