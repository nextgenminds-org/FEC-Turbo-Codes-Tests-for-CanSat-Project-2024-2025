#include <iostream>  // For input/output operations
#include <cmath>     // For mathematical functions (not used in the code)
#include <cstring>   // For string manipulation
#include <limits>    // For defining infinity and numeric limits
#include <vector>    // For dynamic arrays (not directly used here)
#include <cstdlib>   // For general-purpose functions
#include <cstdint>   // For fixed-width integer types

#define MAX_DATA_LENGTH 1024                 // Maximum length of input data
#define INF std::numeric_limits<double>::infinity() // Infinity constant (not used in this code)

// Class to implement convolutional encoding and decoding
template<typename T> class ccode {
public:
    // Constructor to initialize the ccode class with specified parameters
    ccode(uint32_t uint_n, uint32_t uint_k, uint32_t uint_m)
        : uint_n(uint_n), uint_k(uint_k), uint_m(uint_m), uint_mmask((1u << uint_m) - 1),
          uint_gen(new uint32_t[uint_n]), uint_current_state(0) {}

    // Destructor to free dynamically allocated memory
    ~ccode() { delete[] uint_gen; }

    // Method to set generator polynomials for encoding
    void set_generator(uint32_t uint_gen_input, uint32_t uint_gen_index) {
        if (uint_gen_index < uint_n)  // Ensure index is within bounds
            uint_gen[uint_gen_index] = uint_gen_input;
    }

    // Method to encode input into systematic and parity bits
    void encode(const char *input, char *systematic, char *parity1, char *parity2) {
        size_t length = strlen(input); // Get the length of the input string
        for (size_t i = 0; i < length; ++i) {
            bool bit = input[i] - '0'; // Convert character to binary (0 or 1)
            systematic[i] = input[i]; // Copy systematic bit

            // Generate parity bits
            parity1[i] = (next_output(uint_current_state, bit) & 0b1) + '0';
            parity2[i] = (next_output(uint_current_state, !bit) & 0b1) + '0';

            // Update the current state
            uint_current_state = next_state(uint_current_state, bit);
        }

        // Null-terminate the strings
        systematic[length] = '\0';
        parity1[length] = '\0';
        parity2[length] = '\0';
    }

    // Method to decode systematic and parity bits back into the original message
    void decode(const char *systematic, const char *parity1, const char *parity2, char *output) {
        size_t length = strlen(systematic);
        for (size_t i = 0; i < length; ++i) {
            bool sys_bit = systematic[i] - '0';
            bool parity_bit1 = parity1[i] - '0';
            bool parity_bit2 = parity2[i] - '0';

            // Decode the bit by XORing the systematic and parity bits
            output[i] = (sys_bit ^ parity_bit1 ^ parity_bit2) + '0';
        }
        output[length] = '\0'; // Null-terminate the output string
    }

private:
    uint32_t uint_n, uint_k, uint_m, uint_mmask; // Encoder parameters and mask
    uint32_t *uint_gen;                         // Array of generator polynomials
    uint32_t uint_current_state;                // Current state of the encoder

    // Calculate the next state of the encoder based on current state and input
    uint32_t next_state(uint32_t state, bool input) {
        return ((state << 1) | input) & uint_mmask;
    }

    // Calculate the parity output based on current state and input
    uint8_t next_output(uint32_t state, bool input) {
        uint8_t result = 0;
        for (uint32_t i = 0; i < uint_n; ++i) {
            uint32_t temp = state & uint_gen[i]; // AND operation with generator polynomial
            bool parity = 0;
            // Calculate parity by counting 1s in the result
            while (temp) {
                parity ^= temp & 1;
                temp >>= 1;
            }
            result |= (parity << i); // Combine parity bits for all generators
        }
        return result;
    }
};

// Function declarations
void stringToBinary(const char *input, char *binary); // Converts string to binary
void binaryToString(const char *binary, char *output); // Converts binary to string

int main() {
    // Arrays to hold input, encoded, and decoded data
    char input[MAX_DATA_LENGTH] = {0};
    char binaryInput[MAX_DATA_LENGTH * 8] = {0};
    char systematic[MAX_DATA_LENGTH * 8] = {0};
    char parity1[MAX_DATA_LENGTH * 8] = {0};
    char parity2[MAX_DATA_LENGTH * 8] = {0};
    char decodedBinary[MAX_DATA_LENGTH * 8] = {0};
    char decodedOutput[MAX_DATA_LENGTH] = {0};

    char option; // To store user menu choice
    std::cout << "Turbo Encoder/Decoder Simulation" << std::endl;

    // Create an encoder/decoder with parameters (2,1,3)
    ccode encoderDecoder(2, 1, 3);
    encoderDecoder.set_generator(0b111, 0); // Generator polynomial 1
    encoderDecoder.set_generator(0b101, 1); // Generator polynomial 2

    while (true) {
        // Display menu options
        std::cout << "1. Encode" << std::endl;
        std::cout << "2. Decode" << std::endl;
        std::cout << "Select an option: ";
        std::cin >> option;

        if (option == '1') {
            std::cout << "Enter the message to encode: ";
            std::cin.ignore();
            std::cin.getline(input, MAX_DATA_LENGTH - 1);

            // Convert input string to binary
            stringToBinary(input, binaryInput);
            // Encode binary input
            encoderDecoder.encode(binaryInput, systematic, parity1, parity2);

            // Display encoded message
            std::cout << "Encoded Message:" << std::endl;
            std::cout << "Systematic: " << systematic << std::endl;
            std::cout << "Parity 1:   " << parity1 << std::endl;
            std::cout << "Parity 2:   " << parity2 << std::endl;
        } else if (option == '2') {
            // Decode an encoded message
            std::cout << "Enter encoded systematic bits: ";
            std::cin.ignore();
            std::cin.getline(systematic, MAX_DATA_LENGTH * 8 - 1);
            std::cout << "Enter encoded parity1 bits: ";
            std::cin.getline(parity1, MAX_DATA_LENGTH * 8 - 1);
            std::cout << "Enter encoded parity2 bits: ";
            std::cin.getline(parity2, MAX_DATA_LENGTH * 8 - 1);

            // Decode the input bits
            encoderDecoder.decode(systematic, parity1, parity2, decodedBinary);
            // Convert binary back to string
            binaryToString(decodedBinary, decodedOutput);

            // Display decoded message
            std::cout << "Decoded Binary Message: " << decodedBinary << std::endl;
            std::cout << "Decoded Message: " << decodedOutput << std::endl;
        } else {
            std::cout << "Invalid option. Try again." << std::endl;
        }
    }

    return 0;
}

// Converts a string to binary representation
void stringToBinary(const char *input, char *binary) {
    size_t index = 0;
    for (size_t i = 0; i < strlen(input); ++i) {
        for (int j = 7; j >= 0; --j) {
            binary[index++] = ((input[i] >> j) & 1) + '0';
        }
    }
    binary[index] = '\0'; // Null-terminate the binary string
}

// Converts binary to string representation
void binaryToString(const char *binary, char *output) {
    size_t len = strlen(binary) / 8; // Length of the string
    for (size_t i = 0; i < len; ++i) {
        char ch = 0;
        for (int j = 0; j < 8; ++j) {
            ch |= (binary[i * 8 + j] - '0') << (7 - j);
        }
        output[i] = ch;
    }
    output[len] = '\0'; // Null-terminate the output string
}
