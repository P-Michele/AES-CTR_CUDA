#include <iostream>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include "aes.cuh"
#include <filesystem>

using namespace std;

#define BLOCK_SIZE 256
#define AES_BLOCK_SIZE 16
#define BUFFER_SIZE (64 * 1024 * 1024) // 64MB

// Global variable to keep track of the current chunk index
uint64_t chunkIndex = 0;

int main(int argc, char* argv[]) {

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return -1;
    }
    

    auto start = std::chrono::high_resolution_clock::now();
    string path = "/content/" + string(argv[1]);

    //Define constant parameters in GPU constant memory
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

    //Allocate Unified Memory (Both CPU and GPU) for the buffer
    uint8_t* unified_buffer;
    cudaError_t err = cudaMallocManaged(&unified_buffer, 67108864); // Allocate 64MB
    checkCudaError(err, "Failed to allocate unified memory");

    int device = -1;
    cudaGetDevice(&device);
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    location.id = device;

   //Encryption loop
    while (fin.read(reinterpret_cast<char*>(unified_buffer), 67108864) || fin.gcount() > 0) {

        // Get the actual number of bytes readed
        std::streamsize bytesRead = fin.gcount();

        size_t totalAESBlocks = (static_cast<size_t>(bytesRead) + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
        //Compute number of thread blocks needed to process all the data
        int numBlocks = (totalAESBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

        //Sould increase performances due to the fact that the data is already in GPU memory and ready to be processed.
        cudaMemPrefetchAsync(unified_buffer, bytesRead, location, NULL);
    
        // Encrypt in-place using the unified buffer
        encryptCtr<<<numBlocks, BLOCK_SIZE>>>(unified_buffer, unified_buffer, bytesRead, totalAESBlocks, (chunkIndex * (BUFFER_SIZE / AES_BLOCK_SIZE)));
        chunkIndex++;

        // Wait for GPU to finish before accessing the results on the CPU
        err = cudaDeviceSynchronize();
        checkCudaError(err, "Failed to synchronize device after encryption");
        // Write the encrypted data back to the output file
        fout.write(reinterpret_cast<char*>(unified_buffer), bytesRead);
    }

    checkCudaError(err, "Failed to synchronize device");

    // Free the unified memory
    err = cudaFree(unified_buffer);
    checkCudaError(err, "Failed to free unified memory");
    fin.close();
    fout.close();

    // Check needed time to encrypt the file
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Cifrato con successo. Tempo impiegato: " << duration.count() << " ms" << std::endl;

    fout.open(std::filesystem::current_path() / "data.csv", std::ios::app);
    if (fout.is_open()) {
        fout << "GPU Test with file " << path << ", Time: " << duration.count() << " ms" << std::endl;
        fout.close();
    } else {
        std::cerr << "Unable to open data.csv for writing." << std::endl;
    }


    return 0;
}
