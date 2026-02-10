# 1024-QAM

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
    if len(signal.shape) > 1:
        signal_power = np.mean(np.abs(signal)**2, axis=1, keepdims=True)
        if np.isinf(snr_db): return signal
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    else:
        signal_power = np.mean(np.abs(signal)**2)
        if np.isinf(snr_db): return signal
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

def integrate_output(I_out, Q_out, N):
    if len(I_out.shape) > 1:
        num_symbols = I_out.shape[0]
        I_hats = np.zeros(num_symbols)
        Q_hats = np.zeros(num_symbols)
        ones_kernel = np.ones(N, dtype=float)

        for i in range(num_symbols):
            power = I_out[i]**2 + Q_out[i]**2
            energy = np.convolve(power, ones_kernel, mode='valid')
            start = int(np.argmax(energy)) if len(energy) > 0 else 0
            end = min(start + N, len(I_out[i]))
            I_hats[i] = np.sum(I_out[i, start:end])
            Q_hats[i] = np.sum(Q_out[i, start:end])
        return I_hats, Q_hats
    else:
        power = I_out**2 + Q_out**2
        if len(power) < N: return 0, 0
        energy = np.convolve(power, np.ones(N, dtype=float), mode='valid')
        if len(energy) == 0: return 0, 0
        start = int(np.argmax(energy))
        end = start + N
        return np.sum(I_out[start:end]), np.sum(Q_out[start:end])

def bits_to_symbol_indices(bits, bits_per_symbol):
    num_symbols = len(bits) // bits_per_symbol
    bits = bits[:num_symbols * bits_per_symbol]
    bits_reshaped = bits.reshape(-1, bits_per_symbol)
    powers_of_two = 1 << np.arange(bits_per_symbol)[::-1]
    return bits_reshaped @ powers_of_two

def symbol_indices_to_bits(indices, bits_per_symbol):
    n_symbols = len(indices)
    bits = np.zeros((n_symbols, bits_per_symbol), dtype=int)
    for i in range(bits_per_symbol):
        bits[:, i] = (indices >> (bits_per_symbol - 1 - i)) & 1
    return bits.flatten()

def decide_symbol_index(received_points, reference_symbols):
    dists = np.abs(received_points[:, None] - reference_symbols[None, :])
    return np.argmin(dists, axis=1)

def main():
    N = 64
    fs = 640e9; fc = 50e9
    num_lpf_taps = N + 1; lpf_cutoff = 50e9
    Ts = 1 / fs; t_sampled = np.arange(N) * Ts

    bits_per_symbol = 10
    M = 1024

    num_test_symbols = 200000
    total_bits = num_test_symbols * bits_per_symbol
    SNR_dB = 26
    read_noise_stddev_scaled = 0.0025

    g_min = 1e-3; g_max = 8e-3; states = 76; nonlinearity = 0.38

    print(f"--- Simulation Start: 1024-QAM ---")
    print(f"Freq: {fc/1e9}GHz, SNR: {SNR_dB}dB")

    b_coeffs = generate_lpf_coefficients(num_lpf_taps, lpf_cutoff, fs)
    lpf_toeplitz = create_toeplitz_matrix(b_coeffs, N)
    t = np.arange(N) / fs
    cos_carrier = np.cos(2 * np.pi * fc * t)
    sin_carrier_neg = -np.sin(2 * np.pi * fc * t)
    I_demod_ideal = np.diag(cos_carrier) @ lpf_toeplitz
    Q_demod_ideal = np.diag(sin_carrier_neg) @ lpf_toeplitz

    qam1024_points = []
    for i_val in range(-31, 32, 2):
        for q_val in range(-31, 32, 2):
            qam1024_points.append(i_val + 1j * q_val)
    ideal_symbols = np.array(qam1024_points)
    ideal_symbols = ideal_symbols / np.sqrt(np.mean(np.abs(ideal_symbols)**2))

    M_mod_ideal = create_modulation_matrix(ideal_symbols, fc, t_sampled)

    print("Mapping RRAM...")
    min_mod = M_mod_ideal.min(); offset_mod = -min_mod
    target_mod = M_mod_ideal + offset_mod
    max_mod = target_mod.max()
    G_mod_rram = map_weights_to_conductance_linear(target_mod, g_min, g_max, states, nonlinearity, max_mod)

    min_demod = min(I_demod_ideal.min(), Q_demod_ideal.min()); offset_demod = -min_demod
    I_target = I_demod_ideal + offset_demod; Q_target = Q_demod_ideal + offset_demod
    max_demod = max(I_target.max(), Q_target.max())
    G_I_rram = map_weights_to_conductance_linear(I_target, g_min, g_max, states, nonlinearity, max_demod)
    G_Q_rram = map_weights_to_conductance_linear(Q_target, g_min, g_max, states, nonlinearity, max_demod)

    gain_mod = max_mod / (g_max - g_min)
    gain_demod = max_demod / (g_max - g_min)

    cal_sym = ideal_symbols[0]
    cal_sig = np.real(cal_sym)*np.cos(2*np.pi*fc*t) - np.imag(cal_sym)*np.sin(2*np.pi*fc*t)
    I_meas, Q_meas = integrate_output(cal_sig @ I_demod_ideal, cal_sig @ Q_demod_ideal, N)
    scale_factor = (I_meas + 1j * Q_meas) / cal_sym if cal_sym != 0 else 1.0

    tx_bits = np.random.randint(0, 2, total_bits)
    tx_indices = bits_to_symbol_indices(tx_bits, bits_per_symbol)

    print("Processing...")
    mod_raw_batch = G_mod_rram[tx_indices]
    noise_scale_mod = read_noise_stddev_scaled * ((g_max - g_min) / max_mod)
    mod_noisy = mod_raw_batch + np.random.normal(0, noise_scale_mod, mod_raw_batch.shape)
    tx_sig = (mod_noisy - g_min) * gain_mod - offset_mod

    rx_sig = add_awgn(tx_sig, SNR_dB)

    I_out = rx_sig @ G_I_rram
    Q_out = rx_sig @ G_Q_rram
    noise_scale_demod = read_noise_stddev_scaled * ((g_max - g_min) / max_demod)
    I_out += np.random.normal(0, noise_scale_demod, I_out.shape)
    Q_out += np.random.normal(0, noise_scale_demod, Q_out.shape)

    rx_sums = np.sum(rx_sig, axis=1, keepdims=True)
    I_final = (I_out - g_min*rx_sums) * gain_demod - offset_demod*rx_sums
    Q_final = (Q_out - g_min*rx_sums) * gain_demod - offset_demod*rx_sums

    I_vals, Q_vals = integrate_output(I_final, Q_final, N)
    rx_points = (I_vals + 1j * Q_vals) / scale_factor

    rx_indices = decide_symbol_index(rx_points, ideal_symbols)
    rx_bits = symbol_indices_to_bits(rx_indices, bits_per_symbol)

    tx_points_ideal = ideal_symbols[tx_indices]
    evm = np.sqrt(np.mean(np.abs(rx_points - tx_points_ideal)**2)) / np.sqrt(np.mean(np.abs(ideal_symbols)**2)) * 100

    print(f"[Results] EVM: {evm:.2f}%")

    tx_points_ideal = ideal_symbols[tx_indices]
    df_const = pd.DataFrame({
        'Tx_Index': tx_indices,
        'Tx_Real': np.real(tx_points_ideal),
        'Tx_Imag': np.imag(tx_points_ideal),
        'Rx_Real': np.real(rx_points),
        'Rx_Imag': np.imag(rx_points)
    })

    csv_filename = 'rram_1024qam_constellation.csv'
    df_const.to_csv(csv_filename, index=False)
    print(f"Constellation points saved to '{csv_filename}'")

    plt.figure(figsize=(10, 10))
    plt.scatter(np.real(rx_points), np.imag(rx_points), c='blue', s=1, alpha=0.3, label='Rx (RRAM)')
    plt.title(f"1024-QAM (EVM={evm:.2f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == '__main__':
    main()
