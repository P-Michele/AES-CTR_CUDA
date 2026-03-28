#include <iostream>
#include <cmath>
#include "aes.h"
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

#define AES_BLOCK_SIZE 16;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    string path = argv[1];

    //AES key and nonce should be 16 bytes and 8 bytes respectively for AES-128 in CTR mode
    AES aes((const uint8_t*)"thisisakey12345", (const uint8_t*)"nonce123");

    size_t dotPos = path.find_last_of(".");
    string newPath = (dotPos != string::npos) ? string(path).insert(dotPos, "_encrypted") : path + "_encrypted";

    std::ifstream fin(path, std::ios::binary);
    std::ofstream fout(newPath, std::ios::binary);

    if (!fout.is_open() || !fin.is_open()) { 
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    std::vector<uint8_t> buffer(67108864);//64MB buffer size

    while (fin.read(reinterpret_cast<char*>(buffer.data()), buffer.size()) || fin.gcount() > 0) {
        
        std::streamsize bytesRead = fin.gcount();
    
        aes.encryptCtr(buffer.data(), buffer.data(), bytesRead);

        //Print ciphertext chunck
        fout << std::hex << std::setfill('0');

        fout.write(reinterpret_cast<const char*>(buffer.data()), bytesRead);
    }

    fin.close();
    fout.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Cifrato con successo. Tempo impiegato: " << duration.count() << " ms" << std::endl;

    fout.open("C:\\Users\\perin\\Desktop\\data.csv", std::ios::app);
    if (fout.is_open()) {
        fout << "CPU Test with file " << argv[1] << ", Time: " << duration.count() << " ms" << std::endl;
        fout.close();
    }

    return 0;
}
