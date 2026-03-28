#include "aes.h"
#include "box.h"
#include <cmath>
#include <cstdint>
#include <vector>

AES::AES(const uint8_t* key, const uint8_t* nonce) {
    for (int i = 0; i < 16; i++) {
        this->key[i] = key[i];
    }
    for (int i = 0; i < 8; i++) {
        this->nonce[i] = nonce[i];
    }
    keyExpansion();
    this->counterIndex = 0;
}

void AES::encryptCtr(const uint8_t* plaintext, uint8_t* ciphertext, size_t length) {
    size_t roundNumber = (length + 15) / 16;
    uint8_t counter[16];

    // Encrypt loop
    for(size_t i=0; i < roundNumber; i++){
        
        // Counter setup
        for(int j=0; j<8; j++){
            counter[j] = nonce[j];
        }

        counterAssignement(counter, this->counterIndex);
        counterIndex++;

        encryptBlock(counter);

        // XOR the encrypted counter with the plaintext to get the ciphertext block
        size_t offset = i * 16;
        for (int j = 0; j < 16; j++) {
            if (offset + j < length) {
                ciphertext[offset + j] = counter[j] ^ plaintext[offset + j];
            }
        }
    }
}

void AES::encryptBlock(uint8_t* state) {
    addRoundKey(state, roundKeys);

    for (int round = 1; round < 10; round++) {
        subBytes(state);
        shiftRows(state);
        mixColumns(state);
        addRoundKey(state, roundKeys + round * 16);
    }

    subBytes(state);
    shiftRows(state);
    addRoundKey(state, roundKeys + 160);
}


void AES::keyExpansion() {
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
void AES::addRoundKey(uint8_t* state, const uint8_t* roundKey) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}

/* Substitute bytes in the state using the S-box for confusion*/
void AES::subBytes(uint8_t* state) {
    for (int i = 0; i < 16; i++) {
        state[i] = sbox[state[i]];
    }
}

/* Shift rows in the state for diffusion */
void AES::shiftRows(uint8_t* state) {
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


void AES::mixColumns(uint8_t* state) {
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

void AES::counterAssignement(uint8_t* counter, uint64_t block_idx) {
    counter[15] = block_idx & 0xFF;
    counter[14] = (block_idx >> 8) & 0xFF;
    counter[13] = (block_idx >> 16) & 0xFF;
    counter[12] = (block_idx >> 24) & 0xFF;
    counter[11] = (block_idx >> 32) & 0xFF;
    counter[10] = (block_idx >> 40) & 0xFF;
    counter[9] = (block_idx >> 48) & 0xFF;
    counter[8] = (block_idx >> 56) & 0xFF;
}
