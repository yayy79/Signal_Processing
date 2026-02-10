# Hardware-Aware Communication and Signal Processing Simulations Using RRAM Models

This repository contains the simulation codes used to generate the system-level results. The codes implement communication signal processing and spectral analysis tasks using hardware-aware vector–matrix multiplication (VMM) models that explicitly incorporate resistive random-access memory (RRAM) non-idealities.

## Overview

The purpose of this repository is to support **reproducibility, transparency, and validation** of the numerical results presented in the paper. The simulations demonstrate how non-volatile RRAM-based analog VMM can be used to perform high-frequency communication signal processing tasks, including high-order QAM demodulation, discrete Fourier transform (DFT), and multi-input multi-output (MIMO) detection.

All computations are performed with RRAM device characteristics explicitly reflected through finite conductance states, nonlinearity, and read noise.

## Scope of the Simulations

The codes included in this repository cover the following tasks:

### 1. High-Order QAM Demodulation Using RRAM VMM
- 16-QAM
- 64-QAM
- 256-QAM
- 1024-QAM

For each modulation order, the full modulation–demodulation chain is implemented using:
- Carrier-based I/Q modulation
- FIR low-pass filtering represented by Toeplitz matrices
- Offset-based single-array RRAM mapping
- Explicit modeling of conductance quantization, nonlinearity, and read noise
- Performance evaluation using error vector magnitude (EVM)

### 2. RRAM-Based Discrete Fourier Transform (DFT)
- 64-point DFT implemented as a matrix–vector multiplication
- Real and imaginary DFT kernels mapped to physical RRAM conductance matrices
- Multi-tone input signal analysis in the frequency domain
- Comparison between ideal DFT and RRAM-based DFT outputs

### 3. 2×2 MIMO Detection with MMSE Equalization
- 4-QAM modulation
- Random Rayleigh fading channel
- MMSE detection mapped onto an RRAM crossbar
- Complex-to-real matrix expansion for hardware-compatible implementation
- Evaluation of BER, SER, and EVM per antenna

## Hardware-Aware Modeling

All simulations explicitly account for RRAM non-idealities, including:
- Finite conductance range (G_min to G_max)
- Discrete conductance states
- Nonlinear conductance programming behavior
- Additive read noise

An offset-based mapping strategy is used to implement signed weights using a single unipolar RRAM array.

Required Python libraries:
numpy
scipy
matplotlib
pandas

No proprietary software or datasets are required.

Reproducibility Notes
Random symbol streams and noise realizations are used; results may exhibit minor run-to-run variations unless a fixed random seed is specified.
Simulation parameters (e.g., SNR, sampling frequency, number of symbols, RRAM states) are explicitly defined within the code.

Output Data
The simulations automatically generate CSV files
These files can be directly used to regenerate figures or perform further analysis.

Limitations
The simulations focus on algorithmic and system-level validation rather than circuit-level modeling.
Parasitic effects, latency, and energy consumption are not explicitly simulated.
The code is not optimized for execution speed and is intended for research use only.

This repository is provided for:
Reproduction of the numerical results reported in the associated publication
Academic research and education
Extension to related hardware-aware signal processing studies
