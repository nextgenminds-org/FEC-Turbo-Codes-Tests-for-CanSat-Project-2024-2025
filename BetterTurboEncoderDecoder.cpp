#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>

// Utility functions for basic mathematical operations

// Returns the absolute value of a number
template <typename T>
constexpr T abs(T x)
{
    return (x < 0) ? -x : x;
}

// Computes the base-2 logarithm of an integer
template <typename T>
constexpr T log2(T x)
{
    T result = 0;
    while (x >>= 1) // Shifts bits to the right until x becomes 0
        ++result;
    return result;
}

// Computes 2 raised to the power of the given exponent
template <typename T>
constexpr T pow2(T exp)
{
    return 1 << exp; // Bitwise shift left for efficient computation
}

// Convolutional Encoder (ccode) Implementation
class ConvolutionalCode
{
private:
    uint32_t n, m; // Number of output bits and memory size (constraint length)
    uint32_t state; // Current state of the encoder
    std::vector<uint32_t> generators; // Generator polynomials for the encoder

    // Computes the next state of the encoder given the current state and input bit
    uint32_t computeNextState(uint32_t currentState, bool input)
    {
        return ((currentState << 1) | input) & ((1 << m) - 1);
    }

    // Computes the output bits for the current state and input bit
    uint8_t computeNextOutput(uint32_t currentState, bool input)
    {
        uint8_t output = 0; // Encoded output bits
        for (size_t i = 0; i < n; ++i) // Iterate through each generator
        {
            uint32_t temp = currentState & generators[i]; // AND operation with generator
            if (input) // Include input bit in calculation if needed
                temp |= (1 << (m - 1)); // Shift input into position

            // Count the number of 1s in temp (parity calculation)
            output |= (__builtin_popcount(temp) % 2) << i; // Store parity bit
        }
        return output;
    }

public:
    // Constructor initializes the encoder with the number of outputs, memory size, and generator polynomials
    ConvolutionalCode(uint32_t n, uint32_t m, const std::vector<uint32_t> &gen)
        : n(n), m(m), generators(gen), state(0) {}

    // Resets the encoder state to 0
    void reset() { state = 0; }

    // Encodes a binary input sequence and produces systematic and parity outputs
    std::vector<std::pair<uint8_t, uint8_t>> encode(const std::vector<uint8_t> &input)
    {
        std::vector<std::pair<uint8_t, uint8_t>> output; // Systematic and parity pairs
        for (bool bit : input) // Iterate through each input bit
        {
            uint8_t parity = computeNextOutput(state, bit); // Compute parity bits
            state = computeNextState(state, bit); // Update encoder state
            output.emplace_back(bit, parity); // Store systematic and parity bits
        }
        return output;
    }

    // Implements the BCJR algorithm for soft decoding using log-likelihood ratios (LLRs)
    std::vector<double> decodeBCJR(const std::vector<double> &systematic, const std::vector<double> &parity)
    {
        size_t length = systematic.size();
        size_t numStates = 1 << m; // Total number of encoder states
        std::vector<std::vector<double>> alpha(length + 1, std::vector<double>(numStates, -std::numeric_limits<double>::infinity()));
        std::vector<std::vector<double>> beta(length + 1, std::vector<double>(numStates, -std::numeric_limits<double>::infinity()));
        std::vector<double> llr(length, 0.0); // Log-likelihood ratios for each bit

        // Initialize alpha and beta values
        alpha[0][0] = 0.0; // Start in state 0
        beta[length][0] = 0.0; // End in state 0

        // Forward recursion for alpha
        for (size_t t = 0; t < length; ++t)
        {
            for (size_t state = 0; state < numStates; ++state)
            {
                for (bool input : {false, true}) // Iterate over possible input bits
                {
                    size_t nextState = computeNextState(state, input); // Determine next state
                    double gamma = systematic[t] * (2 * input - 1) + parity[t] * (2 * input - 1); // Compute gamma value
                    alpha[t + 1][nextState] = std::max(alpha[t + 1][nextState], alpha[t][state] + gamma);
                }
            }
        }

        // Backward recursion for beta
        for (size_t t = length; t > 0; --t)
        {
            for (size_t state = 0; state < numStates; ++state)
            {
                for (bool input : {false, true}) // Iterate over possible input bits
                {
                    size_t nextState = computeNextState(state, input); // Determine next state
                    double gamma = systematic[t - 1] * (2 * input - 1) + parity[t - 1] * (2 * input - 1); // Compute gamma value
                    beta[t - 1][state] = std::max(beta[t - 1][state], beta[t][nextState] + gamma);
                }
            }
        }

        // Compute LLRs for each bit
        for (size_t t = 0; t < length; ++t)
        {
            double prob0 = -std::numeric_limits<double>::infinity(), prob1 = -std::numeric_limits<double>::infinity();
            for (size_t state = 0; state < numStates; ++state)
            {
                for (bool input : {false, true}) // Iterate over possible input bits
                {
                    size_t nextState = computeNextState(state, input); // Determine next state
                    double gamma = systematic[t] * (2 * input - 1) + parity[t] * (2 * input - 1); // Compute gamma value
                    double metric = alpha[t][state] + gamma + beta[t + 1][nextState]; // Combine alpha, gamma, and beta
                    if (!input)
                        prob0 = std::max(prob0, metric); // Update probability for bit 0
                    else
                        prob1 = std::max(prob1, metric); // Update probability for bit 1
                }
            }
            llr[t] = prob1 - prob0; // LLR is the difference between probabilities
        }

        return llr;
    }
};

// Additional utility functions for string and binary conversions

// Converts a string to a binary vector
std::vector<uint8_t> stringToBinary(const std::string &input)
{
    std::vector<uint8_t> binary;
    for (char c : input) // Iterate through each character in the string
    {
        for (int i = 7; i >= 0; --i) // Extract each bit from the character
        {
            binary.push_back((c >> i) & 1);
        }
    }
    return binary;
}

// Converts a binary vector to a string
std::string binaryToString(const std::vector<uint8_t> &binary)
{
    std::string output;
    for (size_t i = 0; i < binary.size(); i += 8) // Process 8 bits at a time
    {
        char c = 0;
        for (int j = 0; j < 8; ++j) // Combine bits into a character
        {
            c = (c << 1) | binary[i + j];
        }
        output += c; // Append the character to the output string
    }
    return output;
}

// Turbo Codec
class TurboCodec
{
private:
    // Two convolutional encoders used in the Turbo Codec
    ConvolutionalCode encoder1, encoder2;

public:
    // Constructor initializes two convolutional encoders with specific generator polynomials
    TurboCodec()
        : encoder1(2, 3, {0b1011, 0b1111}), // Encoder 1 with rate 1/2 and 3 memory elements
          encoder2(2, 3, {0b1011, 0b1111}) // Encoder 2 with similar configuration
    {}

    // Encodes the input string using Turbo encoding
    void encode(const std::string &input, std::string &output)
    {
        // Convert input string to a binary representation
        auto binaryInput = stringToBinary(input);

        // Encode using the first encoder (systematic and parity bits)
        auto encoded1 = encoder1.encode(binaryInput);

        // Create a permuted version of the input (interleaving step)
        std::vector<uint8_t> permutedInput(binaryInput.rbegin(), binaryInput.rend());

        // Encode the permuted input using the second encoder
        auto encoded2 = encoder2.encode(permutedInput);

        // Generate the final encoded output
        output.clear();
        for (size_t i = 0; i < binaryInput.size(); ++i)
        {
            // Concatenate systematic bit, parity bit from encoder1, and parity bit from encoder2
            output += (encoded1[i].first ? "1" : "0");   // Systematic bit
            output += (encoded1[i].second ? "1" : "0");  // Parity bit from encoder1
            output += (encoded2[i].second ? "1" : "0");  // Parity bit from encoder2
        }
    }

    // Decodes the input Turbo encoded string
    void decode(const std::string &input, std::string &output)
    {
        // Determine the length of the input message
        size_t length = input.size() / 3; // Each bit group has 3 bits (systematic + 2 parity bits)

        // Split the input into systematic bits, parity bits from encoder1, and parity bits from encoder2
        std::vector<double> systematic(length), parity1(length), parity2(length);
        for (size_t i = 0; i < length; ++i)
        {
            systematic[i] = input[i * 3] - '0';     // Extract systematic bits
            parity1[i] = input[i * 3 + 1] - '0';   // Extract parity bits from encoder1
            parity2[i] = input[i * 3 + 2] - '0';   // Extract parity bits from encoder2
        }

        // First decoding pass using BCJR on encoder1
        auto extrinsic1 = encoder1.decodeBCJR(systematic, parity1);

        // Reverse the systematic sequence for the second decoding pass
        std::vector<double> systematicReversed(systematic.rbegin(), systematic.rend());

        // Second decoding pass using BCJR on encoder2
        auto extrinsic2 = encoder2.decodeBCJR(systematicReversed, parity2);

        // Reconstruct the original binary message (hard decision decoding)
        std::vector<uint8_t> reconstructedMessage;
        for (size_t i = 0; i < length; ++i)
        {
            reconstructedMessage.push_back(systematic[i] > 0 ? 1 : 0); // Decide bit based on systematic value
        }

        // Convert the binary message back to a string
        output = binaryToString(reconstructedMessage);
    }
};

// Main function to interact with the Turbo Codec
int main()
{
    TurboCodec codec; // Create a TurboCodec instance
    std::string input, encodedOutput, decodedOutput;
    char option;

    std::cout << "Turbo Encoder/Decoder\n"; // Display program title
    while (true) // Infinite loop to allow multiple operations
    {
        // Display menu options
        std::cout << "1. Encode\n2. Decode\nSelect an option: ";
        std::cin >> option;
        std::cin.ignore(); // Ignore any leftover characters in the input buffer

        if (option == '1') // Encoding option
        {
            std::cout << "Enter the message to encode: ";
            std::getline(std::cin, input); // Get input string from user

            codec.encode(input, encodedOutput); // Encode the input message
            std::cout << "Encoded message: " << encodedOutput << "\n"; // Display encoded message
        }
        else if (option == '2') // Decoding option
        {
            std::cout << "Enter the encoded message: ";
            std::getline(std::cin, encodedOutput); // Get encoded message from user

            codec.decode(encodedOutput, decodedOutput); // Decode the message
            std::cout << "Decoded message: " << decodedOutput << "\n"; // Display decoded message
        }
        else // Invalid option
        {
            std::cout << "Invalid option.\n"; // Display error message
        }
    }

    return 0; // End program
}
