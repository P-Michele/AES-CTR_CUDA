#ifndef AES_CUH
#define AES_CUH
#include <cstdint>
#include <cuda_runtime.h>

// Function prototypes
__host__ int defineConstantParameters();
__global__ void encryptCtr(const uint8_t* plaintext, uint8_t* ciphertext, size_t bytes, size_t totalBlocks, uint64_t chunkIndex);
__host__ void checkCudaError(cudaError_t err, const char* msg);

#endif