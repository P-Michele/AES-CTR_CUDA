#include "aes.cuh"
#include <cmath>
#include <cstdint>
#include <vector>
#include <iostream>

__constant__ uint8_t d_sbox[256];
__constant__ uint8_t d_roundKeys[176];
__constant__ uint8_t nonce[8];

static const uint8_t sbox[256] = {
  //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };

static const uint8_t Rcon[11] = {
  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

uint8_t key[16] = { 'm','y', 's', 'e', 'c', 'r', 'e', 't', 'k', 'e', 'y', '1', '2', '3', '4', '5' };
uint8_t nonce_str[8] = { 'm','y', 'n', 'o', 'n', 'c', 'e', '1' };
uint8_t roundKeys[176];


// Prototypes
__device__ void encryptBlock(uint8_t* state);
__device__ void subBytes(uint8_t* state);
__device__ void shiftRows(uint8_t* state);
__device__ void addRoundKey(uint8_t* state, const uint8_t* roundKey);
__device__ void mixColumns(uint8_t* state);
__device__ void counterAssignement(uint8_t* counter, uint64_t block_idx);
__device__ uint8_t xtime(uint8_t x);
__global__ void encryptCtr(const uint8_t* plaintext, uint8_t* ciphertext, size_t bytes, size_t totalBlocks, uint64_t globalBlockOffset);
__host__ void keyExpansion();
__host__ void setupAESConstants();
__host__ int defineConstantParameters();
__host__ void checkCudaError(cudaError_t err, const char* msg);


/*
 Main encryption function, each thread will encrypt a block of 16 bytes, the counter is setup with the nonce and the block index,
 then we encrypt the counter and XOR it with the plaintext to get the ciphertext.
 */
__global__ void encryptCtr(const uint8_t* plaintext, uint8_t* ciphertext, size_t bytes, size_t totalBlocks, uint64_t globalBlockOffset) {
    size_t workIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (workIndex >= totalBlocks) return;

    uint8_t counter[16];

    for (int j = 0; j < 8; j++) {
        counter[j] = nonce[j];
    }

    counterAssignement(counter, workIndex + globalBlockOffset);
    encryptBlock(counter);

    size_t offset = workIndex * 16;
    
    // Check if we are safely within the bounds for a full 16-byte vectorized operation
    if (offset + 16 <= bytes) {
        // Cast to uint4 (16 bytes) for a single coalesced memory transaction
        const uint4* pt_ptr = reinterpret_cast<const uint4*>(plaintext + offset);
        uint4* ct_ptr = reinterpret_cast<uint4*>(ciphertext + offset);
        uint4* ctr_ptr = reinterpret_cast<uint4*>(counter);

        uint4 pt_val = *pt_ptr;
        uint4 ctr_val = *ctr_ptr;

        uint4 ct_val;
        ct_val.x = pt_val.x ^ ctr_val.x;
        ct_val.y = pt_val.y ^ ctr_val.y;
        ct_val.z = pt_val.z ^ ctr_val.z;
        ct_val.w = pt_val.w ^ ctr_val.w;

        *ct_ptr = ct_val;
    } else {
        // Fallback for the final partial block
        for (int j = 0; j < 16; j++) {
            size_t currentByte = offset + j;
            if (currentByte < bytes) {
                ciphertext[currentByte] = counter[j] ^ plaintext[currentByte];
            }
        }
    }
}

__device__ void encryptBlock(uint8_t* state) {
    addRoundKey(state, d_roundKeys);

    for (int round = 1; round < 10; round++) {
        subBytes(state);
        shiftRows(state);
        mixColumns(state);
        addRoundKey(state, d_roundKeys + round * 16);
    }

    subBytes(state);
    shiftRows(state);
    addRoundKey(state, d_roundKeys + 160);
}

/* Block of 16 byte are dependante by previuos block, so we can not parallize this function */
__host__ void keyExpansion() {
    for (int i = 0; i < 16; i++) {
        roundKeys[i] = key[i];
    }

    for (int i = 16; i < 176; i += 4) {
        uint8_t temp[4];
        for (int j = 0; j < 4; j++) {
            temp[j] = roundKeys[i - 4 + j];
        }

        if (i % 16 == 0) {
            uint8_t t = temp[0];
            temp[0] = sbox[temp[1]] ^ Rcon[i / 16];
            temp[1] = sbox[temp[2]];
            temp[2] = sbox[temp[3]];
            temp[3] = sbox[t];
        }

        for (int j = 0; j < 4; j++) {
            roundKeys[i + j] = roundKeys[i - 16 + j] ^ temp[j];
        }
    }
}

/* Add (Xor) the round key to the state */
__device__ void addRoundKey(uint8_t* state, const uint8_t* roundKey) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

/* Substitute bytes in the state using the S-box for confusion*/
__device__ void subBytes(uint8_t* state) {
    for (int i = 0; i < 16; i++) {
        state[i] = d_sbox[state[i]];
    }
}

/* Shift rows in the state for diffusion */
__device__ void shiftRows(uint8_t* state) {
    uint8_t temp[16];
    for (int i = 0; i < 16; i++) {
        temp[i] = state[i];
    }

    state[1] = temp[5];
    state[5] = temp[9];
    state[9] = temp[13];
    state[13] = temp[1];

    state[2] = temp[10];
    state[6] = temp[14];
    state[10] = temp[2];
    state[14] = temp[6];

    state[3] = temp[15];
    state[7] = temp[3];
    state[11] = temp[7];
    state[15] = temp[11];
}


__device__ void mixColumns(uint8_t* state) {
    uint8_t temp[16];
    for (int i = 0; i < 16; i++) {
        temp[i] = state[i];
    }

    // Process column by column (i steps by 4)
    for (int i = 0; i < 16; i += 4) {
        state[i]     = xtime(temp[i]) ^ (xtime(temp[i+1]) ^ temp[i+1]) ^ temp[i+2] ^ temp[i+3];
        state[i+1]   = temp[i] ^ xtime(temp[i+1]) ^ (xtime(temp[i+2]) ^ temp[i+2]) ^ temp[i+3];
        state[i+2]   = temp[i] ^ temp[i+1] ^ xtime(temp[i+2]) ^ (xtime(temp[i+3]) ^ temp[i+3]);
        state[i+3]   = (xtime(temp[i]) ^ temp[i]) ^ temp[i+1] ^ temp[i+2] ^ xtime(temp[i+3]);
    }
}

/* Assign the block index to the counter (last 8 bytes) */
__device__ void counterAssignement(uint8_t* counter, uint64_t block_idx) {
    counter[15] = block_idx & 0xFF;
    counter[14] = (block_idx >> 8) & 0xFF;
    counter[13] = (block_idx >> 16) & 0xFF;
    counter[12] = (block_idx >> 24) & 0xFF;
    counter[11] = (block_idx >> 32) & 0xFF;
    counter[10] = (block_idx >> 40) & 0xFF;
    counter[9] = (block_idx >> 48) & 0xFF;
    counter[8] = (block_idx >> 56) & 0xFF;
}

__host__ void setupAESConstants() {
    cudaMemcpyToSymbol(d_sbox, sbox, sizeof(sbox));
}

__device__ uint8_t xtime(uint8_t x) {
    return (x << 1) ^ (((x >> 7) & 1) * 0x1b);
}

/*  
Used at the start of AES to define the round keys and copy them to GPU constant memory, also copy the nonce and boxes to GPU constant memory 
*/
__host__ int defineConstantParameters() {

    keyExpansion();
    setupAESConstants();
     //Copy CPU value to GPU constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_roundKeys, roundKeys, 176);
    if (err != cudaSuccess) {
        std::cerr << "Error copying round key to GPU constant memory: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMemcpyToSymbol(nonce, nonce_str, 8);
    if (err != cudaSuccess) {
        std::cerr << "Error copying nonce to GPU constant memory: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    return 0;
}


__host__ void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}