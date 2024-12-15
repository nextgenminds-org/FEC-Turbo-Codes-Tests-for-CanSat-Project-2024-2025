/*
===============================================================================
Turbo Codec BCJR Algorithm Implementation in C++
===============================================================================

### Description:
This program implements a **Turbo Codec**, a sophisticated error-correcting system 
widely used in digital communication systems to ensure data integrity over noisy 
channels. Turbo Codes are highly efficient in detecting and correcting errors, 
making them essential in applications like mobile networks, satellite communication, 
and data storage.

The implementation includes two primary processes:
1. **Encoding**: Converts a message into a highly redundant encoded format that 
   can withstand errors during transmission.
2. **Decoding**: Recovers the original message from the noisy received data using 
   probabilistic methods.

### Key Features:
1. **Convolutional Encoding**:
   - The input message is encoded using two convolutional encoders.
   - The first encoder processes the message as is.
   - The second encoder processes a shuffled (interleaved) version of the message.
   - The output consists of the original data bits (systematic bits) and parity bits 
     from both encoders.

2. **Interleaving**:
   - A pseudo-random interleaver shuffles the order of bits before they are processed 
     by the second encoder. This improves error-correction performance by reducing 
     correlation between errors.

3. **Decoding with BCJR Algorithm**:
   - The **BCJR algorithm** is used for decoding. It is a probabilistic algorithm 
     that computes the **Log-Likelihood Ratios (LLRs)** for each bit in the message.
   - The BCJR algorithm performs forward and backward recursions to calculate 
     the likelihood of each bit being 0 or 1 based on the received data.
   - The decoder iteratively refines these probabilities using information from 
     both convolutional decoders until convergence.

4. **Iterative Decoding**:
   - The decoding process alternates between the two convolutional decoders, 
     refining the LLRs at each step. This iterative approach ensures high decoding 
     accuracy, even in noisy environments.

### Workflow:
1. **Encoding**:
   - The user inputs a message, which is converted into binary format.
   - The binary message is encoded by two convolutional encoders, with interleaving 
     applied before the second encoder.
   - The encoded message includes the original bits and parity bits, output as a 
     single binary string.

2. **Decoding**:
   - The user inputs the encoded binary string.
   - The decoder splits the string into systematic bits and parity bits from both 
     encoders.
   - Using the BCJR algorithm, the decoder iteratively computes LLRs for each bit 
     and reconstructs the most likely original message.

### How to Use:
- 1. Compile the program using a C++ compiler (e.g., `g++`):
   ```bash
   g++ -o turbo_codec turbo_codec.cpp
- Run the compiled program:

./turbo_codec

- Follow the prompts:
    - To encode a message, select option 1 and enter the text.
    - To decode a message, select option 2 and enter the encoded string.

## Example:
- Input Message: Hello
- Encoded Message: A long binary string containing systematic and parity bits.
- Decoded Message: Hello (retrieved even if the encoded message is slightly corrupted).

## BCJR Algorithm:
The BCJR algorithm (named after its creators Bahl, Cocke, Jelinek, and Raviv) is a probabilistic decoding algorithm that operates in the logarithmic domain to efficiently compute LLRs for each bit. The algorithm consists of:

- Forward Recursion: Calculates the likelihood of reaching a particular state at each time step, based on the previous states.
- Backward Recursion: Computes the likelihood of transitioning from a state at a given time step to the final state.
- LLR Computation: Combines the forward and backward probabilities to estimate the most likely value (0 or 1) for each bit.

The BCJR algorithm is central to the iterative decoding process, enabling the Turbo Codec to achieve high error-correction performance.

=============================================================================== */



#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

// Utility functions

// Template function to compute the absolute value of a number.
// Works for both integer and floating-point types.
template <typename T>
constexpr T abs(T x)
{
    return (x < 0) ? -x : x;
}

// Function to generate a pseudo-random interleaver.
// An interleaver shuffles the indices of the input sequence to enhance error-correction performance.
// `length`: Number of elements to shuffle.
// Returns a vector of shuffled indices.
std::vector<size_t> generateInterleaver(size_t length)
{
    std::vector<size_t> indices(length); // Initialize a vector with sequential indices [0, 1, ..., length-1].
    for (size_t i = 0; i < length; ++i)
        indices[i] = i;

    std::mt19937 generator(42); // Use a fixed seed for reproducibility of results.
    std::shuffle(indices.begin(), indices.end(), generator); // Shuffle the indices.
    return indices;
}

// Class implementing a convolutional encoder.
class ConvolutionalCode
{
private:
    uint32_t n, m; // Number of output bits (`n`) and memory size (`m`).
    uint32_t state; // Current state of the encoder (stored in `m` bits).
    std::vector<uint32_t> generators; // Generator polynomials defining the encoder.

    // Compute the next state of the encoder given the current state and input bit.
    uint32_t computeNextState(uint32_t currentState, bool input)
    {
        return ((currentState << 1) | input) & ((1 << m) - 1); // Shift left, add input, and mask to keep `m` bits.
    }

    // Compute the output parity bits based on the current state and input bit.
    uint8_t computeNextOutput(uint32_t currentState, bool input)
    {
        uint8_t output = 0; // Initialize output to 0.
        for (size_t i = 0; i < n; ++i) // Iterate through each generator polynomial.
        {
            uint32_t temp = currentState & generators[i]; // Mask current state with the generator polynomial.
            if (input) // If the input bit is 1, include the highest bit of the state.
                temp |= (1 << (m - 1));
            output |= (__builtin_popcount(temp) % 2) << i; // Compute the parity bit for this generator.
        }
        return output; // Return the computed parity bits.
    }

public:
    // Constructor: Initialize the encoder with the number of outputs (`n`), memory size (`m`), and generator polynomials.
    ConvolutionalCode(uint32_t n, uint32_t m, const std::vector<uint32_t> &gen)
        : n(n), m(m), generators(gen), state(0) {}

    // Reset the encoder state to 0.
    void reset() { state = 0; }

    // Encode a binary input sequence and return the systematic and parity bits as pairs.
    std::vector<std::pair<uint8_t, uint8_t>> encode(const std::vector<uint8_t> &input)
    {
        std::vector<std::pair<uint8_t, uint8_t>> output; // Vector to store the encoded pairs.
        for (bool bit : input) // Process each input bit.
        {
            uint8_t parity = computeNextOutput(state, bit); // Compute the parity bits.
            state = computeNextState(state, bit); // Update the encoder state.
            output.emplace_back(bit, parity); // Append the systematic bit and parity bits as a pair.
        }
        return output; // Return the encoded sequence.
    }

    // Decode using the BCJR algorithm.
    // This function computes log-likelihood ratios (LLRs) for each bit based on the input probabilities.
    std::vector<double> decodeBCJR(const std::vector<double> &systematic, const std::vector<double> &parity, const std::vector<double> &extrinsic)
    {
        size_t length = systematic.size(); // Length of the input sequence.
        size_t numStates = 1 << m; // Total number of states in the encoder (2^m).
        
        // Initialize forward (alpha) and backward (beta) probabilities with -infinity (logarithmic domain).
        std::vector<std::vector<double>> alpha(length + 1, std::vector<double>(numStates, -std::numeric_limits<double>::infinity()));
        std::vector<std::vector<double>> beta(length + 1, std::vector<double>(numStates, -std::numeric_limits<double>::infinity()));
        std::vector<double> llr(length, 0.0); // Log-likelihood ratios for each bit.

        alpha[0][0] = 0.0; // Initial state probability for forward recursion.
        beta[length][0] = 0.0; // Initial state probability for backward recursion.

        // Forward recursion: Compute alpha probabilities.
        for (size_t t = 0; t < length; ++t)
        {
            for (size_t state = 0; state < numStates; ++state)
            {
                for (bool input : {false, true}) // Evaluate both possible inputs (0 and 1).
                {
                    size_t nextState = computeNextState(state, input); // Compute the next state.
                    double gamma = systematic[t] * (2 * input - 1) + parity[t] * (2 * input - 1) + extrinsic[t] * (2 * input - 1);
                    alpha[t + 1][nextState] = std::max(alpha[t + 1][nextState], alpha[t][state] + gamma);
                }
            }
        }

        // Backward recursion: Compute beta probabilities.
        for (size_t t = length; t > 0; --t)
        {
            for (size_t state = 0; state < numStates; ++state)
            {
                for (bool input : {false, true}) // Evaluate both possible inputs (0 and 1).
                {
                    size_t nextState = computeNextState(state, input); // Compute the next state.
                    double gamma = systematic[t - 1] * (2 * input - 1) + parity[t - 1] * (2 * input - 1) + extrinsic[t - 1] * (2 * input - 1);
                    beta[t - 1][state] = std::max(beta[t - 1][state], beta[t][nextState] + gamma);
                }
            }
        }

        // Compute log-likelihood ratios (LLRs) for each bit.
        for (size_t t = 0; t < length; ++t)
        {
            double prob0 = -std::numeric_limits<double>::infinity(), prob1 = -std::numeric_limits<double>::infinity();
            for (size_t state = 0; state < numStates; ++state)
            {
                for (bool input : {false, true})
                {
                    size_t nextState = computeNextState(state, input); // Compute the next state.
                    double gamma = systematic[t] * (2 * input - 1) + parity[t] * (2 * input - 1) + extrinsic[t] * (2 * input - 1);
                    double metric = alpha[t][state] + gamma + beta[t + 1][nextState];
                    if (!input)
                        prob0 = std::max(prob0, metric); // Update probability for input = 0.
                    else
                        prob1 = std::max(prob1, metric); // Update probability for input = 1.
                }
            }
            llr[t] = prob1 - prob0; // Compute LLR as the difference of probabilities in log domain.
        }

        return llr; // Return the computed LLRs.
    }
};

// Helper functions to convert between strings and binary representations.

// Convert a string to a binary representation (vector of 0s and 1s).
std::vector<uint8_t> stringToBinary(const std::string &input)
{
    std::vector<uint8_t> binary;
    for (char c : input)
    {
        for (int i = 7; i >= 0; --i) // Extract each bit from the character.
        {
            binary.push_back((c >> i) & 1);
        }
    }
    return binary;
}

// Convert a binary representation (vector of 0s and 1s) back to a string.
std::string binaryToString(const std::vector<uint8_t> &binary)
{
    std::string output;
    for (size_t i = 0; i < binary.size(); i += 8) // Process each group of 8 bits.
    {
        char c = 0;
        for (int j = 0; j < 8; ++j)
        {
            c = (c << 1) | binary[i + j];
        }
        output += c;
    }
    return output;
}

// Turbo Codec class: Combines the functionalities of encoding and decoding.
class TurboCodec
{
private:
    ConvolutionalCode encoder1, encoder2; // Two convolutional encoders.

public:
    // Constructor: Initialize the two convolutional encoders with predefined generator polynomials.
    TurboCodec()
        : encoder1(2, 3, {0b1011, 0b1111}), encoder2(2, 3, {0b1011, 0b1111}) {}

    // Encode a string message into a Turbo Code format.
    void encode(const std::string &input, std::string &output)
    {
        auto binaryInput = stringToBinary(input); // Convert the input string to binary.

        auto interleaver = generateInterleaver(binaryInput.size()); // Generate an interleaver.

        auto encoded1 = encoder1.encode(binaryInput); // Encode the original binary input.

        // Apply the interleaver to the input.
        std::vector<uint8_t> permutedInput(binaryInput.size());
        for (size_t i = 0; i < binaryInput.size(); ++i)
            permutedInput[interleaver[i]] = binaryInput[i];

        auto encoded2 = encoder2.encode(permutedInput); // Encode the permuted input.

        // Combine the outputs from both encoders into a single string.
        output.clear();
        for (size_t i = 0; i < binaryInput.size(); ++i)
        {
            output += (encoded1[i].first ? "1" : "0"); // Add systematic bit.
            output += (encoded1[i].second ? "1" : "0"); // Add parity bit from first encoder.
            output += (encoded2[i].second ? "1" : "0"); // Add parity bit from second encoder.
        }
    }

    // Decode a Turbo Code formatted string back to the original message.
    void decode(const std::string &input, std::string &output)
    {
        size_t length = input.size() / 3; // Determine the number of bits in the original message.
        std::vector<double> systematic(length), parity1(length), parity2(length);

        // Split the input string into systematic and parity bits.
        for (size_t i = 0; i < length; ++i)
        {
            systematic[i] = input[i * 3] - '0'; // Extract systematic bit.
            parity1[i] = input[i * 3 + 1] - '0'; // Extract parity bit from first encoder.
            parity2[i] = input[i * 3 + 2] - '0'; // Extract parity bit from second encoder.
        }

        std::vector<double> extrinsic1(length, 0.0), extrinsic2(length, 0.0);
        std::vector<double> previousLLR(length, 0.0);

        int maxIterations = 10;             // Maximum number of decoding iterations.
        double convergenceThreshold = 0.01; // Threshold for convergence of LLRs.

        for (int iteration = 0; iteration < maxIterations; ++iteration)
        {
            // Decode using the first encoder and compute extrinsic information.
            extrinsic1 = encoder1.decodeBCJR(systematic, parity1, extrinsic2);
            
            // Permute the extrinsic information from the first decoder.
            std::vector<double> permutedExtrinsic1(extrinsic1.rbegin(), extrinsic1.rend());
            
            // Decode using the second encoder and compute extrinsic information.
            extrinsic2 = encoder2.decodeBCJR(systematic, parity2, permutedExtrinsic1);

            // Check for convergence by comparing the current LLRs with the previous ones.
            double maxDifference = 0.0;
            for (size_t i = 0; i < length; ++i)
            {
                double difference = std::abs(extrinsic1[i] - previousLLR[i]);
                maxDifference = std::max(maxDifference, difference);
                previousLLR[i] = extrinsic1[i]; // Update the previous LLRs.
            }

            if (maxDifference < convergenceThreshold)
            {
                std::cout << "Convergence achieved at iteration " << iteration + 1 << "\n";
                break; // Stop if the LLRs have converged.
            }
        }

        // Convert the LLRs to binary representation by thresholding.
        std::vector<uint8_t> reconstructedMessage(length);
        for (size_t i = 0; i < length; ++i)
        {
            reconstructedMessage[i] = systematic[i] > 0 ? 1 : 0;
        }

        output = binaryToString(reconstructedMessage); // Convert the binary representation back to a string.
    }
};

int main()
{
    TurboCodec codec; // Create an instance of the Turbo Codec.
    std::string input, encodedOutput, decodedOutput;
    char option;

    std::cout << "Turbo Encoder/Decoder\n";
    while (true)
    {
        // Prompt the user to choose an operation.
        std::cout << "1. Encode\n2. Decode\nSelect an option: ";
        std::cin >> option;
        std::cin.ignore(); // Ignore newline character after the option.

        if (option == '1') // Encoding option.
        {
            std::cout << "Enter the message to encode: ";
            std::getline(std::cin, input);

            codec.encode(input, encodedOutput); // Encode the input message.
            std::cout << "Encoded message: " << encodedOutput << "\n";
        }
        else if (option == '2') // Decoding option.
        {
            std::cout << "Enter the encoded message: ";
            std::getline(std::cin, encodedOutput);

            codec.decode(encodedOutput, decodedOutput); // Decode the input message.
            std::cout << "Decoded message: " << decodedOutput << "\n";
        }
        else // Invalid option.
        {
            std::cout << "Invalid option.\n";
        }
    }

    return 0;
}
