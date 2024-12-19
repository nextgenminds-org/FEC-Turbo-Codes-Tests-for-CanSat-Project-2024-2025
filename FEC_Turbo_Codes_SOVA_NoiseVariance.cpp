#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

// Utility functions
template <typename T>
constexpr T abs(T x) {
    return (x < 0) ? -x : x;
}

// Pseudo-random interleaver
std::vector<size_t> generateInterleaver(size_t length) {
    std::vector<size_t> indices(length);
    for (size_t i = 0; i < length; ++i)
        indices[i] = i;

    std::mt19937 generator(42); // Fixed seed for repeatability
    std::shuffle(indices.begin(), indices.end(), generator);
    return indices;
}

// Convolutional Encoder Implementation
class ConvolutionalCode {
private:
    uint32_t n, m; // Outputs and memory size
    uint32_t state;
    std::vector<uint32_t> generators;

    uint32_t computeNextState(uint32_t currentState, bool input) {
        return ((currentState << 1) | input) & ((1 << m) - 1);
    }

    uint8_t computeNextOutput(uint32_t currentState, bool input) {
        uint8_t output = 0;
        for (size_t i = 0; i < n; ++i) {
            uint32_t temp = currentState & generators[i];
            if (input)
                temp |= (1 << (m - 1));
            output |= (__builtin_popcount(temp) % 2) << i; // Parity bit
        }
        return output;
    }

public:
    ConvolutionalCode(uint32_t n, uint32_t m, const std::vector<uint32_t> &gen)
        : n(n), m(m), generators(gen), state(0) {}

    void reset() { state = 0; }

    std::vector<std::pair<uint8_t, uint8_t>> encode(const std::vector<uint8_t> &input) {
        std::vector<std::pair<uint8_t, uint8_t>> output;
        for (bool bit : input) {
            uint8_t parity = computeNextOutput(state, bit);
            state = computeNextState(state, bit);
            output.emplace_back(bit, parity);
        }
        return output;
    }

    std::vector<double> decodeSOVA(const std::vector<double> &systematic,
                                   const std::vector<double> &parity,
                                   const std::vector<double> &extrinsic,
                                   double noiseVariance) {
        size_t length = systematic.size();
        size_t numStates = 1 << m;

        // Path metrics and decisions
        std::vector<double> pathMetrics(numStates, -std::numeric_limits<double>::infinity());
        std::vector<std::vector<int>> decisions(length, std::vector<int>(numStates, -1));
        pathMetrics[0] = 0.0; // Initialize the starting state

        // Forward pass
        for (size_t t = 0; t < length; ++t) {
            std::vector<double> tempPathMetrics(numStates, -std::numeric_limits<double>::infinity());

            for (size_t state = 0; state < numStates; ++state) {
                for (bool input : {false, true}) {
                    size_t nextState = computeNextState(state, input);
                    double gamma = (systematic[t] * (2 * input - 1) +
                                    parity[t] * (2 * input - 1) +
                                    extrinsic[t] * (2 * input - 1)) / noiseVariance;

                    double metric = pathMetrics[state] + gamma;
                    if (metric > tempPathMetrics[nextState]) {
                        tempPathMetrics[nextState] = metric;
                        decisions[t][nextState] = input;
                    }
                }
            }

            pathMetrics = tempPathMetrics;
        }

        // Traceback to find most likely path
        std::vector<int> mostLikelyPath(length, 0);
        size_t state = std::distance(pathMetrics.begin(), std::max_element(pathMetrics.begin(), pathMetrics.end()));

        for (size_t t = length; t > 0; --t) {
            mostLikelyPath[t - 1] = decisions[t - 1][state];
            state = computeNextState(state, mostLikelyPath[t - 1]);
        }

        // Calculate LLR
        std::vector<double> llr(length, 0.0);
        for (size_t t = 0; t < length; ++t) {
            double metric0 = -std::numeric_limits<double>::infinity();
            double metric1 = -std::numeric_limits<double>::infinity();

            for (size_t state = 0; state < numStates; ++state) {
                for (bool input : {false, true}) {
                    size_t nextState = computeNextState(state, input);
                    double gamma = (systematic[t] * (2 * input - 1) +
                                    parity[t] * (2 * input - 1) +
                                    extrinsic[t] * (2 * input - 1)) / noiseVariance;

                    double metric = pathMetrics[state] + gamma;
                    if (!input)
                        metric0 = std::max(metric0, metric);
                    else
                        metric1 = std::max(metric1, metric);
                }
            }

            llr[t] = metric1 - metric0;
        }

        return llr;
    }
};

// Helper functions
std::vector<uint8_t> stringToBinary(const std::string &input) {
    std::vector<uint8_t> binary;
    for (char c : input) {
        for (int i = 7; i >= 0; --i) {
            binary.push_back((c >> i) & 1);
        }
    }
    return binary;
}

std::string binaryToString(const std::vector<uint8_t> &binary) {
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

// Turbo Codec
class TurboCodec {
private:
    ConvolutionalCode encoder1, encoder2;

public:
    TurboCodec()
        : encoder1(2, 3, {0b1011, 0b1111}), encoder2(2, 3, {0b1011, 0b1111}) {}

    void encode(const std::string &input, std::string &output) {
        auto binaryInput = stringToBinary(input);

        // Generate interleaver
        auto interleaver = generateInterleaver(binaryInput.size());

        auto encoded1 = encoder1.encode(binaryInput);

        // Apply interleaver
        std::vector<uint8_t> permutedInput(binaryInput.size());
        for (size_t i = 0; i < binaryInput.size(); ++i)
            permutedInput[interleaver[i]] = binaryInput[i];

        auto encoded2 = encoder2.encode(permutedInput);

        output.clear();
        for (size_t i = 0; i < binaryInput.size(); ++i) {
            output += (encoded1[i].first ? "1" : "0");
            output += (encoded1[i].second ? "1" : "0");
            output += (encoded2[i].second ? "1" : "0");
        }
    }

    void decode(const std::string &input, std::string &output, double noiseVariance) {
        size_t length = input.size() / 3;
        std::vector<double> systematic(length), parity1(length), parity2(length);

        for (size_t i = 0; i < length; ++i) {
            systematic[i] = input[i * 3] - '0';
            parity1[i] = input[i * 3 + 1] - '0';
            parity2[i] = input[i * 3 + 2] - '0';
        }

        std::vector<double> extrinsic1(length, 0.0), extrinsic2(length, 0.0);
        int maxIterations = 20;
        double convergenceThreshold = 0.001;

        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            extrinsic1 = encoder1.decodeSOVA(systematic, parity1, extrinsic2, noiseVariance);

            std::vector<double> permutedExtrinsic1(length);
            for (size_t i = 0; i < length; ++i) {
                permutedExtrinsic1[i] = extrinsic1[i];
            }

            extrinsic2 = encoder2.decodeSOVA(systematic, parity2, permutedExtrinsic1, noiseVariance);

            double maxDifference = 0.0;
            for (size_t i = 0; i < length; ++i) {
                maxDifference = std::max(maxDifference, std::abs(extrinsic1[i] - systematic[i]));
            }
            if (maxDifference < convergenceThreshold) {
                break;
            }
        }

        std::vector<uint8_t> reconstructedMessage(length);
        for (size_t i = 0; i < length; ++i) {
            reconstructedMessage[i] = systematic[i] > 0 ? 1 : 0;
        }

        output = binaryToString(reconstructedMessage);
    }
};

int main() {
    TurboCodec codec;
    std::string input, encodedOutput, decodedOutput;
    char option;
    double noiseVariance = 0.5;

    std::cout << "Turbo Encoder/Decoder\n";
    while (true) {
        std::cout << "1. Encode\n2. Decode\n3. Set Noise Variance (current: " << noiseVariance << ")\nSelect an option: ";
        std::cin >> option;
        std::cin.ignore();

        if (option == '1') {
            std::cout << "Enter the message to encode: ";
            std::getline(std::cin, input);
            codec.encode(input, encodedOutput);
            std::cout << "Encoded message: " << encodedOutput << "\n";
        } else if (option == '2') {
            std::cout << "Enter the encoded message: ";
            std::getline(std::cin, encodedOutput);
            codec.decode(encodedOutput, decodedOutput, noiseVariance);
            std::cout << "Decoded message: " << decodedOutput << "\n";
        } else if (option == '3') {
            std::cout << "Enter new noise variance: ";
            std::cin >> noiseVariance;
            std::cin.ignore();
            std::cout << "Noise variance updated to " << noiseVariance << "\n";
        } else {
            std::cout << "Invalid option.\n";
        }
    }

    return 0;
}