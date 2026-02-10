# 64-QAM

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import firwin

def map_weights_to_conductance_linear(weights_matrix, g_min, g_max, num_states, non_linearity, max_weight_for_scaling):
    if num_states <= 1: return np.full_like(weights_matrix, g_min)
    p_levels = np.linspace(0, 1, num_states)
    alpha = 1 + non_linearity
    p_nonlinear = p_levels ** alpha
    achievable_conductances_lut = g_min + p_nonlinear * (g_max - g_min)

    if max_weight_for_scaling == 0: return np.full_like(weights_matrix, g_min)


    scaled_matrix = g_min + (weights_matrix / max_weight_for_scaling) * (g_max - g_min)
    scaled_matrix = np.clip(scaled_matrix, g_min, g_max)

    rram_matrix = np.zeros_like(scaled_matrix)
    for i in range(scaled_matrix.shape[0]):
        for j in range(scaled_matrix.shape[1]):
            target_val = scaled_matrix[i, j]
            closest_idx = np.argmin(np.abs(achievable_conductances_lut - target_val))
            rram_matrix[i, j] = achievable_conductances_lut[closest_idx]

    return rram_matrix

def generate_lpf_coefficients(num_taps, cutoff_freq, sampling_freq):
    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff = cutoff_freq / nyquist_freq
    return firwin(num_taps, normalized_cutoff, window='hamming')

def create_toeplitz_matrix(coeffs, num_inputs):
    num_taps = len(coeffs)
    output_len = num_inputs + num_taps - 1
    toeplitz_matrix = np.zeros((num_inputs, output_len))
    for i in range(num_inputs):
        toeplitz_matrix[i, i:i+num_taps] = coeffs
    return toeplitz_matrix[:, :2*num_inputs]

def create_modulation_matrix(ideal_symbols, fc, t_sampled):
    N = len(t_sampled)
    num_symbols = len(ideal_symbols)
    mod_matrix = np.zeros((num_symbols, N))

    cos_carrier = np.cos(2 * np.pi * fc * t_sampled)
    sin_carrier = np.sin(2 * np.pi * fc * t_sampled)

    for i, symbol in enumerate(ideal_symbols):
        I_sym = np.real(symbol)
        Q_sym = np.imag(symbol)
        mod_matrix[i, :] = I_sym * cos_carrier - Q_sym * sin_carrier

    return mod_matrix

def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    if np.isinf(snr_db): return signal
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

def integrate_output(I_out, Q_out, N):
    """복조된 파형을 적분하여 좌표로 변환"""
    power = I_out**2 + Q_out**2
    if len(power) < N: return 0, 0
    energy = np.convolve(power, np.ones(N, dtype=float), mode='valid')
    if len(energy) == 0: return 0, 0
    start = int(np.argmax(energy))
    end = start + N
    I_hat = np.sum(I_out[start:end])
    Q_hat = np.sum(Q_out[start:end])
    return I_hat, Q_hat

def bits_to_symbol_indices(bits, bits_per_symbol):
    num_symbols = len(bits) // bits_per_symbol
    bits = bits[:num_symbols * bits_per_symbol]
    bits_reshaped = bits.reshape(-1, bits_per_symbol)
    powers_of_two = 1 << np.arange(bits_per_symbol)[::-1]
    symbol_indices = bits_reshaped @ powers_of_two
    return symbol_indices

def symbol_indices_to_bits(indices, bits_per_symbol):
    bits_list = []
    for idx in indices:
        binary_string = format(idx, f'0{bits_per_symbol}b')
        bits = [int(b) for b in binary_string]
        bits_list.extend(bits)
    return np.array(bits_list)

def decide_symbol_index(received_point, reference_symbols):
    distances = np.abs(received_point - reference_symbols)
    closest_idx = np.argmin(distances)
    return closest_idx

def gray_to_binary(n):
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return n

def main():
    N = 64
    fs = 1066.7e9
    fc = 50e9
    num_lpf_taps = N + 1
    lpf_cutoff = 50e9

    Ts = 1 / fs; t_sampled = np.arange(N) * Ts

    bits_per_symbol = 6; M = 64

    num_test_symbols = 200000
    total_bits = num_test_symbols * bits_per_symbol

    SNR_dB = 10; read_noise_stddev_scaled = 0.0025

    g_min = 1e-3; g_max = 8e-3; states = 76; nonlinearity = 0.38

    print(f"--- Simulation Start: 64-QAM (Offset Method - 10GHz) ---")
    print(f"Frequency: {fc/1e9} GHz (Carrier), {fs/1e9} GHz (Sampling)")
    print(f"Total Bits: {total_bits}, SNR: {SNR_dB} dB, RRAM Noise: {read_noise_stddev_scaled*100}%")

    b_coeffs = generate_lpf_coefficients(num_lpf_taps, lpf_cutoff, fs)
    lpf_toeplitz = create_toeplitz_matrix(b_coeffs, N)

    t = np.arange(N) / fs
    cos_carrier = np.cos(2 * np.pi * fc * t)
    sin_carrier_neg = -np.sin(2 * np.pi * fc * t)

    I_demod_ideal = np.diag(cos_carrier) @ lpf_toeplitz
    Q_demod_ideal = np.diag(sin_carrier_neg) @ lpf_toeplitz


    L = 8
    k_half = 3
    mask_half = 0x7

    ideal_symbols = []
    for i in range(M):
        i_bits_gray = (i >> k_half) & mask_half
        q_bits_gray = i & mask_half

        i_idx = gray_to_binary(i_bits_gray)
        q_idx = gray_to_binary(q_bits_gray)

        I_val = 2 * i_idx - 7
        Q_val = -(2 * q_idx - 7)

        avg_power = 42.0
        symbol_complex = (I_val + 1j * Q_val) / np.sqrt(avg_power)
        ideal_symbols.append(symbol_complex)

    ideal_symbols = np.array(ideal_symbols)

    M_mod_ideal = create_modulation_matrix(ideal_symbols, fc, t_sampled)

    print("\nMapping Matrices to RRAM (Offset Method)...")

    min_mod = M_mod_ideal.min()
    offset_mod = -min_mod
    target_mod = M_mod_ideal + offset_mod
    max_mod = target_mod.max()
    G_mod_rram = map_weights_to_conductance_linear(target_mod, g_min, g_max, states, nonlinearity, max_mod)

    min_demod = min(I_demod_ideal.min(), Q_demod_ideal.min())
    offset_demod = -min_demod
    I_target = I_demod_ideal + offset_demod
    Q_target = Q_demod_ideal + offset_demod
    max_demod = max(I_target.max(), Q_target.max())

    G_I_rram = map_weights_to_conductance_linear(I_target, g_min, g_max, states, nonlinearity, max_demod)
    G_Q_rram = map_weights_to_conductance_linear(Q_target, g_min, g_max, states, nonlinearity, max_demod)

    gain_mod = max_mod / (g_max - g_min)
    gain_demod = max_demod / (g_max - g_min)

    print("Mapping complete.\n")

    cal_sym = ideal_symbols[0]
    cal_sig = np.real(cal_sym)*np.cos(2*np.pi*fc*t) - np.imag(cal_sym)*np.sin(2*np.pi*fc*t)
    cal_I_ideal = cal_sig @ I_demod_ideal
    cal_Q_ideal = cal_sig @ Q_demod_ideal
    I_meas, Q_meas = integrate_output(cal_I_ideal, cal_Q_ideal, N)
    cal_res = I_meas + 1j * Q_meas
    scale_factor = cal_res / cal_sym if cal_res != 0 else 1.0
    print(f"Scale Factor: {scale_factor:.4f}\n")

    tx_bits = np.random.randint(0, 2, total_bits)
    tx_indices = bits_to_symbol_indices(tx_bits, bits_per_symbol)

    rx_points_rram = []
    rx_points_ideal = []

    print("Transmitting Random Stream...")

    for idx in tx_indices:
        v_in = np.zeros(len(ideal_symbols)); v_in[idx] = 1.0

        tx_sig_ideal = v_in @ M_mod_ideal
        rx_sig_ideal = tx_sig_ideal
        I_out_ideal = rx_sig_ideal @ I_demod_ideal
        Q_out_ideal = rx_sig_ideal @ Q_demod_ideal
        I_val_ideal, Q_val_ideal = integrate_output(I_out_ideal, Q_out_ideal, N)
        rx_points_ideal.append((I_val_ideal + 1j*Q_val_ideal) / scale_factor)

        mod_raw = v_in @ G_mod_rram

        gain_inv_mod = (g_max - g_min) / max_mod
        mod_noise = np.random.normal(0, read_noise_stddev_scaled * gain_inv_mod, mod_raw.shape)
        mod_noisy = mod_raw + mod_noise

        g_min_bias_mod = g_min * np.sum(v_in)
        offset_bias_mod = offset_mod * np.sum(v_in)
        tx_sig_rram = (mod_noisy - g_min_bias_mod) * gain_mod - offset_bias_mod

        rx_sig_rram = add_awgn(tx_sig_rram, SNR_dB)

        I_out_raw = rx_sig_rram @ G_I_rram
        Q_out_raw = rx_sig_rram @ G_Q_rram

        gain_inv_demod = (g_max - g_min) / max_demod
        read_std_demod = read_noise_stddev_scaled * gain_inv_demod
        I_out_raw += np.random.normal(0, read_std_demod, I_out_raw.shape)
        Q_out_raw += np.random.normal(0, read_std_demod, Q_out_raw.shape)

        rx_sum = np.sum(rx_sig_rram)
        g_min_bias_demod = g_min * rx_sum
        offset_bias_demod = offset_demod * rx_sum

        I_final = (I_out_raw - g_min_bias_demod) * gain_demod - offset_bias_demod
        Q_final = (Q_out_raw - g_min_bias_demod) * gain_demod - offset_bias_demod

        I_val, Q_val = integrate_output(I_final, Q_final, N)
        rx_points_rram.append((I_val + 1j*Q_val) / scale_factor)

    rx_points_rram = np.array(rx_points_rram)
    rx_points_ideal = np.array(rx_points_ideal)

    print("Analysis...")
    rx_indices = [decide_symbol_index(p, ideal_symbols) for p in rx_points_rram]
    rx_bits = symbol_indices_to_bits(rx_indices, bits_per_symbol)


    tx_ideal = ideal_symbols[tx_indices]

    evm = np.sqrt(np.mean(np.abs(rx_points_rram - rx_points_ideal)**2)) / np.sqrt(np.mean(np.abs(ideal_symbols)**2)) * 100

    print(f"\n[Results] 64-QAM (Offset Method - Single Array)")
    print(f"EVM: {evm:.2f}%")

    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(rx_points_rram), np.imag(rx_points_rram), c='r', s=5, alpha=0.3, label='RRAM')

    plt.title(f"64-QAM RRAM \nEVM={evm:.2f}%")
    plt.grid(True, alpha=0.3); plt.axis('equal'); plt.legend()
    plt.show()

    # Save
    df = pd.DataFrame({
        'Tx_Idx': tx_indices, 'Rx_Idx': rx_indices,
        'Tx_I': np.real(tx_ideal), 'Tx_Q': np.imag(tx_ideal),
        'Rx_Ideal_I': np.real(rx_points_ideal), 'Rx_Ideal_Q': np.imag(rx_points_ideal),
        'Rx_RRAM_I': np.real(rx_points_rram), 'Rx_RRAM_Q': np.imag(rx_points_rram)
    })
    df.to_csv('rram_offset_64qam_results.csv', index=False)
    print("Saved to rram_offset_64qam_results.csv")

if __name__ == '__main__':
    main()
