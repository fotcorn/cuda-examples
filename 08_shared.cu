#include <cuda_runtime.h>
#include <iostream>

__global__ void map(const int *in, int *out, const size_t n) {
    extern __shared__ int shared[];
    const size_t idx = threadIdx.x;

    if (idx < n) {
        shared[idx] = in[idx] + 10;
    }
    __syncthreads();

    if (idx < n) {
        out[idx] = shared[idx];
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
    int hostIn[n];
    int hostOut[n];
    for (size_t i = 0; i < n; i++) {
        hostIn[i] = i;
    }

    // Allocate memory on the device for both input and output arrays, which have the same size.
    int *deviceIn, *deviceOut;
    CUDA_CHECK(cudaMalloc((void **)&deviceIn, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&deviceOut, n * sizeof(int)));

    // Copy the input array from the host to the device.
    CUDA_CHECK(cudaMemcpy(deviceIn, hostIn, n * sizeof(int), cudaMemcpyHostToDevice));

    map<<<1, n, n * sizeof(int)>>>(deviceIn, deviceOut, n);
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
    CUDA_CHECK(cudaFree(deviceIn));
    CUDA_CHECK(cudaFree(deviceOut));

    return 0;
}
