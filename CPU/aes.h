#pragma once
#include <cstdint>
#include <stddef.h>

class AES {
    public:
        AES(const uint8_t* key, const uint8_t* nonce);

        void encryptCtr(const uint8_t* plaintext, uint8_t* ciphertext, size_t length);
        void decryptCtr(const uint8_t* ciphertext, uint8_t* plaintext, size_t length) {
            encryptCtr(ciphertext, plaintext, length); // Decrypting by encrypting the ciphertext again in CTR mode
        }
        
    private:
        uint8_t key[16];
        uint8_t nonce[8];
        uint8_t roundKeys[176];
        size_t counterIndex;

        void keyExpansion();
        void addRoundKey(uint8_t* state, const uint8_t* roundKey); //state is current block, roundKey is the key for the current round
        void subBytes(uint8_t* state);
        void shiftRows(uint8_t* state);
        void mixColumns(uint8_t* state);
        void encryptBlock(uint8_t* state); 
        void counterAssignement(uint8_t* counter, uint64_t block_idx);
       
        inline uint8_t xtime(uint8_t x) {
            return (x << 1) ^ (((x >> 7) & 1) * 0x1b);
        }
};