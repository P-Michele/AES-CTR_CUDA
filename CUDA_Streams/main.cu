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
#define BUFFER_SIZE (64 * 1024 * 1024) 
#define NUM_STREAMS 2

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

    size_t bufferSize = BUFFER_SIZE;
    
    cudaStream_t streams[NUM_STREAMS];
    uint8_t* h_buffer[NUM_STREAMS];
    uint8_t* d_buffer[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        
        cudaError_t err = cudaMallocHost<uint8_t>((void**)&h_buffer[i], bufferSize);
        checkCudaError(err, "Failed to allocate host memory");
        
        err = cudaMalloc((void**)&d_buffer[i], bufferSize);
        checkCudaError(err, "Failed to allocate device memory");
    }

    int sIdx = 0;
    size_t bytesRead[NUM_STREAMS] = {0, 0};

    fin.read(reinterpret_cast<char*>(h_buffer[0]), bufferSize);
    bytesRead[0] = fin.gcount();

    while (bytesRead[sIdx] > 0) {
        int nextIdx = (sIdx + 1) % NUM_STREAMS;

        size_t totalBlocks = (bytesRead[sIdx] + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
        int numBlocks = (totalBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cudaMemcpyAsync(d_buffer[sIdx], h_buffer[sIdx], bytesRead[sIdx], cudaMemcpyHostToDevice, streams[sIdx]);
        
        encryptCtr<<<numBlocks, BLOCK_SIZE, 0, streams[sIdx]>>>(d_buffer[sIdx], d_buffer[sIdx], bytesRead[sIdx], totalBlocks, (chunkIndex * (BUFFER_SIZE / AES_BLOCK_SIZE)));
        
        cudaMemcpyAsync(h_buffer[sIdx], d_buffer[sIdx], bytesRead[sIdx], cudaMemcpyDeviceToHost, streams[sIdx]);

        chunkIndex++;

        fin.read(reinterpret_cast<char*>(h_buffer[nextIdx]), bufferSize);
        bytesRead[nextIdx] = fin.gcount();

        cudaStreamSynchronize(streams[sIdx]);
        
        fout.write(reinterpret_cast<char*>(h_buffer[sIdx]), bytesRead[sIdx]);

        sIdx = nextIdx;
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(h_buffer[i]);
        cudaFree(d_buffer[i]);
    }

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