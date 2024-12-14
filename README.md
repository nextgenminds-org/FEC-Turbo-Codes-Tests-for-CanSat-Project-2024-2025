# Turbo Codes for Forward Error Correction (FEC)
# NGM Team's Tests for CanSat Project 2024 - 2025

This repository contains implementations of **Turbo Codes**, a powerful class of error correction codes widely used in modern communication systems to ensure reliable data transmission over noisy channels. The project provides multiple decoding algorithms, each tailored to specific performance requirements, offering a balance between accuracy, speed, and computational complexity.

## Files Included

### 1. `FEC_Turbo_Codes_BCJR.cpp`
- Implements the **BCJR (MAP)** decoding algorithm. (this implementation rely heavily on the work that has been done in the Susa Framework <a href="https://libsusa.github.io/" target="_blank">Susa Framework</a>)
- Uses forward (`alpha`) and backward (`beta`) recursions to compute **Log-Likelihood Ratios (LLRs)** for optimal decoding.
- **Key Features**:
  - High accuracy, ideal for noisy environments.
  - Computationally intensive due to recursive probability calculations.
- **Use Case**: Systems requiring precise decoding in high-noise conditions.

### 2. `FEC_Turbo_Codes_MAP.cpp`
- Implements the **MAP (Maximum A Posteriori)** decoding algorithm.
- Focused on accurate LLR computations, similar to BCJR but optimized for certain scenarios.
- **Key Features**:
  - Robust performance in noise-intensive channels.
  - High computational demand due to exhaustive probability evaluation.
- **Use Case**: Environments where decoding accuracy is critical, and computational resources are sufficient.

### 3. `FEC_Turbo_Codes_Hybrid_MAP_SOVA.cpp`
- Combines **MAP** and **SOVA** algorithms in a hybrid approach.
- Starts with MAP decoding for precise results in early iterations, switching to SOVA for faster convergence in subsequent steps.
- **Key Features**:
  - Balances decoding accuracy and computational efficiency.
  - Includes a convergence check based on LLR differences.
- **Use Case**: Ideal for moderate noise conditions where both speed and accuracy are required.

### 4. `FEC_Turbo_Codes_SOVA.cpp`
- Implements the **Soft-Output Viterbi Algorithm (SOVA)** for decoding.
- Focuses on identifying the most likely path through the trellis and computing LLRs for error correction.
- **Key Features**:
  - Faster than MAP-based algorithms, with lower computational overhead.
  - Includes traceback logic to reconstruct the most likely transmitted sequence.
- **Use Case**: Real-time systems or applications with constrained processing power and latency requirements.

### 5. Main Implementation (SOVA-based)
- Implements the **Turbo Codec** structure with convolutional encoding, pseudo-random interleaving, and iterative SOVA decoding.
- **Key Features**:
  - Lightweight and practical for real-world use.
  - Convergence monitoring to stop iterations when the solution stabilizes.
- **Use Case**: Practical systems with moderate error rates and limited computational capacity.

## Features
- **Interleaving**: Implements pseudo-random interleaving to improve performance by decorrelating input errors.
- **Convolutional Encoding**: Supports configurable convolutional encoders with flexible generator polynomials.
- **Iterative Decoding**: All implementations use iterative refinement with configurable maximum iterations and convergence thresholds.
- **Flexible Algorithms**: Offers BCJR (MAP), MAP, SOVA, and a hybrid MAP-SOVA approach to suit diverse requirements.

## Applications
- **Wireless Communication Systems** (LTE, 5G)
- **Satellite and IoT Networks**
- **Space Communications** (e.g., deep-space missions)
- **Data Transmission Systems** requiring **Forward Error Correction (FEC)**

## How to Use
1. Compile any of the provided files using a C++ compiler (`g++`, `clang++`, etc.).
2. Follow the input/output instructions in each file for encoding and decoding.
3. Adjust parameters (e.g., number of iterations, generator polynomials) to optimize for your specific system requirements.

Explore and experiment with Turbo Codes to understand their robustness and versatility in error correction!

## Bibliography

### Books
1. **Error Control Coding: Fundamentals and Applications**  
   Shu Lin, Daniel J. Costello  
   - ISBN: 978-0132837965  

2. **Digital Communication**  
   John G. Proakis, Masoud Salehi  
   - ISBN: 978-0072957167  

3. **Modern Coding Theory**  
   Tom Richardson, Rudiger Urbanke  
   - ISBN: 978-0521852296  

4. **Coding and Information Theory**  
   Steven Roman  
   - ISBN: 978-0387947044  

### Papers
1. **"Near Shannon Limit Error-Correcting Coding and Decoding: Turbo Codes"**  
   Claude Berrou, Alain Glavieux, Punya Thitimajshima (1993)  
   - DOI: 10.1109/ICC.1993.397441  

2. **"Maximum-Likelihood Sequence Estimation of Digital Sequences in the Presence of Intersymbol Interference"**  
   L.R. Bahl, J. Cocke, F. Jelinek, J. Raviv (1974)  
   - DOI: 10.1109/TIT.1974.1055186  

3. **"Iterative Decoding of Binary Block and Convolutional Codes"**  
   Joachim Hagenauer, Elke Offer, Lutz Papke (1996)  
   - DOI: 10.1109/26.539767  

4. **"Turbo Decoding for Satellite and Mobile Communication Systems"**  
   David Divsalar, Frank Pollara (1995)  
   - Published by *Jet Propulsion Laboratory (NASA)*  

### Online Resources
1. **[MIT OpenCourseWare - Principles of Digital Communication](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-450-principles-of-digital-communication-i-fall-2006/)**  
   - Lecture notes and video lectures covering Turbo Codes and decoding algorithms.

2. **[Turbo Codes Tutorial (MathWorks)](https://www.mathworks.com/help/comm/ug/turbo-codes.html)**  
   - Practical guide to implementing Turbo Codes in MATLAB.

3. **[NASA Turbo Code Overview](https://descanso.jpl.nasa.gov/)**  
   - Discusses Turbo Code applications in deep-space communication.

### Standards and Specifications
1. **3GPP LTE Standards**  
   - Refer to 3GPP TS 36.212 for detailed encoding and decoding algorithms.  
   - [3GPP Standards Documentation](https://www.3gpp.org/)

2. **CCSDS Standards**  
   - CCSDS 131.0-B-2, "TM Synchronization and Channel Coding".  
   - [CCSDS Standards](https://public.ccsds.org/)
