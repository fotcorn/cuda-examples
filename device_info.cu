#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "=== Device Information ===" << std::endl;
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Multi-Processor Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Concurrent Kernels: " << prop.concurrentKernels << std::endl;

    std::cout << "\n=== Memory Information ===" << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
    std::cout << "Total Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "Reserved Shared Memory per Block: " << prop.reservedSharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Shared Memory per Block (Opt-in): " << prop.sharedMemPerBlockOptin << " bytes" << std::endl;

    std::cout << "\n=== Thread and Block Information ===" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per Multi-Processor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Threads Dimension: " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max Blocks per Multi-Processor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Max Grid Size: " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;

    std::cout << "\n=== Register Information ===" << std::endl;
    std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Registers per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;

    return 0;
}
