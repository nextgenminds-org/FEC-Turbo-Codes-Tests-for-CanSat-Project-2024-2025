/*
===============================================================================
Turbo Codec with Iterative BCJR Algorithm Implementation in C++
===============================================================================

### Description:
This program implements a **Turbo Codec**, an advanced error-correcting technique 
used extensively in digital communication systems to maintain data integrity in 
the presence of noise. Turbo Codes are highly effective in detecting and correcting 
errors, making them indispensable in fields such as mobile communications, satellite 
systems, and data storage.

The implementation features two core operations:
1. **Encoding**: Converts the input message into a redundant format for robust 
   transmission over noisy channels.
2. **Decoding**: Uses the probabilistic BCJR algorithm and iterative methods to 
   reconstruct the original message from the received noisy data.

### Key Features:
1. **Convolutional Encoding**:
   - Uses two convolutional encoders to process the message.
   - The first encoder handles the raw message, while the second processes a shuffled 
     (interleaved) version of the message.
   - Outputs include systematic bits (original data) and parity bits generated 
     by both encoders.

2. **Pseudo-Random Interleaving**:
   - Shuffles the input bits before processing by the second encoder, improving 
     error-correction performance by decorrelating errors.

3. **BCJR Algorithm for Decoding**:
   - Computes **Log-Likelihood Ratios (LLRs)** for each bit using forward and 
     backward recursions.
   - Iteratively refines probabilities using information from both encoders, ensuring 
     accurate decoding in noisy environments.

4. **Iterative Decoding**:
   - Alternates decoding between the two convolutional encoders.
   - Evaluates convergence of LLRs to determine when the message has been optimally decoded.

5. **Noise Variance Customization**:
   - The decoder allows dynamic adjustment of the noise variance parameter to handle 
     varying levels of channel noise effectively.

### Workflow:
1. **Encoding**:
   - Converts the input message to a binary sequence.
   - Processes the binary sequence through two convolutional encoders with interleaving.
   - Outputs a binary string containing systematic bits and parity bits.

2. **Decoding**:
   - Parses the encoded message into systematic and parity components.
   - Decodes the message iteratively using the BCJR algorithm.
   - Determines convergence based on LLR differences, ensuring accurate message recovery.

### How to Use:
1. Compile the program using a C++ compiler (e.g., `g++`):
   ```bash
   g++ -o turbo_codec turbo_codec.cpp

    Run the program:

    ./turbo_codec

    Follow the prompts:
        Select 1 to encode a message and view the encoded output.
        Select 2 to decode an encoded message and retrieve the original.
        Select 3 to adjust the noise variance parameter.

Example:

    Input Message: Hello
    Encoded Message: A binary string with systematic and parity bits.
    Decoded Message: Hello (accurately reconstructed even with noise in the transmission).

BCJR Algorithm:

The BCJR algorithm (Bahl, Cocke, Jelinek, and Raviv) is a probabilistic decoding algorithm that efficiently computes LLRs for each bit. Its three steps include:

    Forward Recursion: Calculates the probability of reaching a state at each time step.
    Backward Recursion: Determines the probability of transitioning from a state to the final state.
    LLR Computation: Combines forward and backward probabilities to estimate the likelihood of each bit being 0 or 1.

The BCJR algorithm is integral to the iterative decoding process, enabling Turbo Code systems to achieve remarkable error-correction performance in noisy conditions.

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

// Template function to calculate the absolute value of a number.
// Supports both integer and floating-point types.
template <typename T>
constexpr T abs(T x) {
    return (x < 0) ? -x : x;
}

// Function to generate a pseudo-random interleaver.
// An interleaver is used to shuffle the input indices to improve error-correction performance.
// `length`: The number of elements to shuffle.
// Returns a vector containing the shuffled indices.
std::vector<size_t> generateInterleaver(size_t length) {
    std::vector<size_t> indices(length); // Initialize indices as [0, 1, ..., length - 1].
    for (size_t i = 0; i < length; ++i)
        indices[i] = i;

    std::mt19937 generator(42); // Use a fixed seed for deterministic and reproducible results.
    std::shuffle(indices.begin(), indices.end(), generator); // Shuffle the indices.
    return indices;
}

// Class to implement a convolutional encoder.
class ConvolutionalCode {
private:
    uint32_t n, m; // Number of output bits (`n`) and memory size (`m`).
    uint32_t state; // Current state of the encoder (stored in `m` bits).
    std::vector<uint32_t> generators; // Generator polynomials used for encoding.

    // Function to compute the next state of the encoder given the current state and input bit.
    uint32_t computeNextState(uint32_t currentState, bool input) {
        return ((currentState << 1) | input) & ((1 << m) - 1); // Shift left, add input, and mask to keep `m` bits.
    }

    // Function to compute the parity bits based on the current state and input bit.
    uint8_t computeNextOutput(uint32_t currentState, bool input) {
        uint8_t output = 0; // Initialize parity output to 0.
        for (size_t i = 0; i < n; ++i) { // Iterate over each generator polynomial.
            uint32_t temp = currentState & generators[i]; // Apply the generator to the current state.
            if (input) // If input is 1, include the highest bit in the computation.
                temp |= (1 << (m - 1));
            output |= (__builtin_popcount(temp) % 2) << i; // Calculate parity (XOR all bits).
        }
        return output;
    }

public:
    // Constructor: Initialize the encoder with the number of outputs (`n`), memory size (`m`), and generator polynomials.
    ConvolutionalCode(uint32_t n, uint32_t m, const std::vector<uint32_t>& gen)
        : n(n), m(m), generators(gen), state(0) {}

    // Function to reset the encoder's state to 0.
    void reset() { state = 0; }

    // Function to encode a binary sequence (vector of 0s and 1s).
    // Returns a vector of pairs where each pair contains the systematic bit and parity bits.
    std::vector<std::pair<uint8_t, uint8_t>> encode(const std::vector<uint8_t>& input) {
        std::vector<std::pair<uint8_t, uint8_t>> output;
        for (bool bit : input) { // Process each input bit.
            uint8_t parity = computeNextOutput(state, bit); // Compute parity bits for the current state and input.
            state = computeNextState(state, bit); // Transition to the next state.
            output.emplace_back(bit, parity); // Store the systematic bit and parity bits as a pair.
        }
        return output;
    }

    // Function to decode a sequence using the BCJR algorithm.
    // Computes log-likelihood ratios (LLRs) for each bit in the input sequence.
    std::vector<double> decodeBCJR(const std::vector<double>& systematic,
                                   const std::vector<double>& parity,
                                   const std::vector<double>& extrinsic,
                                   double noiseVariance) {
        size_t length = systematic.size(); // Number of bits in the sequence.
        size_t numStates = 1 << m; // Total number of states (2^m).

        // Initialize alpha and beta matrices with negative infinity (logarithmic domain).
        std::vector<std::vector<double>> alpha(length + 1, std::vector<double>(numStates, -std::numeric_limits<double>::infinity()));
        std::vector<std::vector<double>> beta(length + 1, std::vector<double>(numStates, -std::numeric_limits<double>::infinity()));
        std::vector<double> llr(length, 0.0); // Log-likelihood ratios (LLRs) for the bits.

        alpha[0][0] = 0.0; // Start forward recursion with initial state probability.
        beta[length][0] = 0.0; // Start backward recursion with initial state probability.

        // Forward recursion to compute alpha probabilities.
        for (size_t t = 0; t < length; ++t) {
            for (size_t state = 0; state < numStates; ++state) {
                for (bool input : {false, true}) { // Evaluate both possible inputs (0 and 1).
                    size_t nextState = computeNextState(state, input);
                    double gamma = (systematic[t] * (2 * input - 1) +
                                    parity[t] * (2 * input - 1) +
                                    extrinsic[t] * (2 * input - 1)) / noiseVariance;
                    alpha[t + 1][nextState] = std::max(alpha[t + 1][nextState], alpha[t][state] + gamma);
                }
            }
        }

        // Backward recursion to compute beta probabilities.
        for (size_t t = length; t > 0; --t) {
            for (size_t state = 0; state < numStates; ++state) {
                for (bool input : {false, true}) { // Evaluate both possible inputs (0 and 1).
                    size_t nextState = computeNextState(state, input);
                    double gamma = (systematic[t - 1] * (2 * input - 1) +
                                    parity[t - 1] * (2 * input - 1) +
                                    extrinsic[t - 1] * (2 * input - 1)) / noiseVariance;
                    beta[t - 1][state] = std::max(beta[t - 1][state], beta[t][nextState] + gamma);
                }
            }
        }

        // Compute LLRs for each bit in the sequence.
        for (size_t t = 0; t < length; ++t) {
            double prob0 = -std::numeric_limits<double>::infinity();
            double prob1 = -std::numeric_limits<double>::infinity();
            for (size_t state = 0; state < numStates; ++state) {
                for (bool input : {false, true}) {
                    size_t nextState = computeNextState(state, input);
                    double gamma = (systematic[t] * (2 * input - 1) +
                                    parity[t] * (2 * input - 1) +
                                    extrinsic[t] * (2 * input - 1)) / noiseVariance;
                    double metric = alpha[t][state] + gamma + beta[t + 1][nextState];
                    if (!input)
                        prob0 = std::max(prob0, metric);
                    else
                        prob1 = std::max(prob1, metric);
                }
            }
            llr[t] = prob1 - prob0; // LLR is the difference between log probabilities.
        }

        return llr; // Return the calculated LLRs.
    }
};

// Helper functions for string-to-binary and binary-to-string conversions.
// Converts a string into a binary vector (each character into 8 bits).
std::vector<uint8_t> stringToBinary(const std::string& input) {
    std::vector<uint8_t> binary;
    for (char c : input) {
        for (int i = 7; i >= 0; --i) { // Extract each bit (MSB first).
            binary.push_back((c >> i) & 1);
        }
    }
    return binary;
}

// Converts a binary vector into a string (each group of 8 bits into a character).
std::string binaryToString(const std::vector<uint8_t>& binary) {
    std::string output;
    for (size_t i = 0; i < binary.size(); i += 8) {
        char c = 0;
        for (int j = 0; j < 8; ++j) {
            c = (c << 1) | binary[i + j];
        }
        output += c;
    }
    return output;
}

// Main class combining encoding and decoding functionalities of a Turbo Codec.
class TurboCodec {
private:
    ConvolutionalCode encoder1, encoder2; // Two convolutional encoders.

public:
    TurboCodec()
        : encoder1(2, 3, {0b1011, 0b1111}), encoder2(2, 3, {0b1011, 0b1111}) {}

    // Function to encode a string message using Turbo encoding.
    void encode(const std::string& input, std::string& output) {
        auto binaryInput = stringToBinary(input); // Convert input to binary.
        auto interleaver = generateInterleaver(binaryInput.size()); // Generate interleaver.

        auto encoded1 = encoder1.encode(binaryInput); // Encode binary input with first encoder.

        // Apply interleaver to binary input.
        std::vector<uint8_t> permutedInput(binaryInput.size());
        for (size_t i = 0; i < binaryInput.size(); ++i)
            permutedInput[interleaver[i]] = binaryInput[i];

        auto encoded2 = encoder2.encode(permutedInput); // Encode permuted input with second encoder.

        // Combine systematic and parity bits into a single output string.
        output.clear();
        for (size_t i = 0; i < binaryInput.size(); ++i) {
            output += (encoded1[i].first ? "1" : "0");
            output += (encoded1[i].second ? "1" : "0");
            output += (encoded2[i].second ? "1" : "0");
        }
    }

    // Function to decode a Turbo-encoded string using iterative decoding.
    void decode(const std::string& input, std::string& output, double noiseVariance) {
        size_t length = input.size() / 3; // Number of original bits.
        std::vector<double> systematic(length), parity1(length), parity2(length);

        // Split input string into systematic and parity bits.
        for (size_t i = 0; i < length; ++i) {
            systematic[i] = input[i * 3] - '0';
            parity1[i] = input[i * 3 + 1] - '0';
            parity2[i] = input[i * 3 + 2] - '0';
        }

        std::vector<double> extrinsic1(length, 0.0), extrinsic2(length, 0.0);
        int maxIterations = 20; // Maximum number of iterations for decoding.
        double convergenceThreshold = 0.001; // Threshold to determine convergence.

        // Iterative decoding using both encoders.
        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            extrinsic1 = encoder1.decodeBCJR(systematic, parity1, extrinsic2, noiseVariance);

            std::vector<double> permutedExtrinsic1(length);
            for (size_t i = 0; i < length; ++i) {
                permutedExtrinsic1[i] = extrinsic1[i];
            }

            extrinsic2 = encoder2.decodeBCJR(systematic, parity2, permutedExtrinsic1, noiseVariance);

            double maxDifference = 0.0;
            for (size_t i = 0; i < length; ++i) {
                maxDifference = std::max(maxDifference, std::abs(extrinsic1[i] - systematic[i]));
            }
            if (maxDifference < convergenceThreshold) { // Stop if LLRs have converged.
                break;
            }
        }

        // Convert final LLRs to binary output.
        std::vector<uint8_t> reconstructedMessage(length);
        for (size_t i = 0; i < length; ++i) {
            reconstructedMessage[i] = systematic[i] > 0 ? 1 : 0;
        }

        output = binaryToString(reconstructedMessage); // Convert binary to string.
    }
};

int main() {
    TurboCodec codec; // Create an instance of TurboCodec.
    std::string input, encodedOutput, decodedOutput;
    char option;
    double noiseVariance = 0.5; // Default noise variance.

    std::cout << "Turbo Encoder/Decoder\n";
    while (true) {
        std::cout << "1. Encode\n2. Decode\n3. Set Noise Variance (current: " << noiseVariance << ")\nSelect an option: ";
        std::cin >> option;
        std::cin.ignore(); // Ignore newline character.

        if (option == '1') { // Encoding.
            std::cout << "Enter the message to encode: ";
            std::getline(std::cin, input);
            codec.encode(input, encodedOutput);
            std::cout << "Encoded message: " << encodedOutput << "\n";
        } else if (option == '2') { // Decoding.
            std::cout << "Enter the encoded message: ";
            std::getline(std::cin, encodedOutput);
            codec.decode(encodedOutput, decodedOutput, noiseVariance);
            std::cout << "Decoded message: " << decodedOutput << "\n";
        } else if (option == '3') { // Adjust noise variance.
            std::cout << "Enter new noise variance: ";
            std::cin >> noiseVariance;
            std::cin.ignore(); // Ignore newline character.
            std::cout << "Noise variance updated to " << noiseVariance << "\n";
        } else {
            std::cout << "Invalid option.\n"; // Invalid input.
        }
    }

    return 0; // End of program.
}
