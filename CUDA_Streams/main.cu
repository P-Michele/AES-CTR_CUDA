#include <iostream>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include "aes.cuh"

using namespace std;

#define BLOCK_SIZE 256
#define AES_BLOCK_SIZE 16
#define BUFFER_SIZE (64 * 1024 * 1024) // 64MB

uint64_t chunkIndex = 0;

int main(int argc, char* argv[]) {

    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    string path = "/content/" + string(argv[1]);

    if(defineConstantParameters() != 0) {
        return -1;
    }

    size_t dotPos = path.find_last_of(".");
    string newPath = (dotPos != string::npos) ? string(path).insert(dotPos, "_encrypted") : path + "_encrypted";

    std::ofstream fout(newPath, std::ios::binary);
    std::ifstream fin(path, std::ios::binary);

    if (!fin) {
        std::cerr << "Error opening file: " << path << std::endl;
        return -1;
    }

    size_t bufferSize = 67108864;
    uint8_t* h_buffer;
    uint8_t* d_buffer;

    // Allocate Pinned memory to speedup data transfer between CPU and GPU
    cudaError_t err = cudaMallocHost<uint8_t>((void**)&h_buffer, bufferSize);
    checkCudaError(err, "Failed to allocate host memory");

    err = cudaMalloc((void**)&d_buffer, bufferSize);
    checkCudaError(err, "Failed to allocate device memory");

    while (fin.read(reinterpret_cast<char*>(h_buffer), bufferSize) || fin.gcount() > 0) {
        std::streamsize bytesRead = fin.gcount();
        //Compute number of 16 bytes blocks
        size_t totalBlocks = (static_cast<size_t>(bytesRead) + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
        //Compute number of thread blocks needed to process all the data
        int numBlocks = (totalBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

        err = cudaMemcpy(d_buffer, h_buffer, bytesRead, cudaMemcpyHostToDevice);
        checkCudaError(err, "Failed to copy data to GPU");

        encryptCtr<<<numBlocks, BLOCK_SIZE>>>(d_buffer, d_buffer, bytesRead, totalBlocks, (chunkIndex * (BUFFER_SIZE / AES_BLOCK_SIZE)));
        chunkIndex++;

        err = cudaMemcpy(h_buffer, d_buffer, bytesRead, cudaMemcpyDeviceToHost);
        checkCudaError(err, "Failed to copy data back to CPU");

        fout.write(reinterpret_cast<char*>(h_buffer), bytesRead);
    }

    cudaFreeHost(h_buffer);
    cudaFree(d_buffer);
    fin.close();
    fout.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Cifrato con successo. Tempo impiegato: " << duration.count() << " ms" << std::endl;

    fout.open("/content/data.csv", std::ios::app);
    if (fout.is_open()) {
        fout << "GPU Test with file " << path << ", Time: " << duration.count() << " ms" << std::endl;
        fout.close();
    } else {
        std::cerr << "Unable to open data.csv for writing." << std::endl;
    }

    return 0;
}