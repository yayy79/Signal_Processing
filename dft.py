# DFT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def main():
    N = 64
    fs = 160e9
    Ts = 1 / fs

    g_min_siemens = 1e-3
    g_max_siemens = 8e-3
    effective_states = 76
    nonlinearity_param = 0.38
    read_noise_std_scaled = 0.0025

    print("--- RRAM DFT Simulation (160GHz Band) ---")
    print(f"N (DFT Size): {N}")
    print(f"Sampling Freq: {fs/1e9:.1f} GHz")
    print(f"\n--- RRAM Device Parameters ---")
    print(f"Conductance: {g_min_siemens*1e3:.1f} ~ {g_max_siemens*1e3:.1f} mS")
    print(f"States: {effective_states}, Nonlinearity (v): {nonlinearity_param}\n")


    t_sampled = np.arange(N) * Ts
    f1, f2, f3 = 12e9, 24.0e9, 55.0e9


    raw_signal = (np.sin(2 * np.pi * f1 * t_sampled) +
                  0.8 * np.sin(2 * np.pi * f2 * t_sampled) +
                  0.3 * np.sin(2 * np.pi * f3 * t_sampled))
    input_vector = raw_signal / np.max(np.abs(raw_signal))


    n = np.arange(N)
    k = n.reshape((N, 1))
    WN = np.exp(-2j * np.pi * k * n / N)

    M_real_ideal = np.real(WN)
    M_imag_ideal = np.imag(WN)


    print("Mapping weights to RRAM (Offset Method)...")
    min_val = min(M_real_ideal.min(), M_imag_ideal.min())
    offset_D = -min_val
    M_real_target = M_real_ideal + offset_D
    M_imag_target = M_imag_ideal + offset_D


    max_target_val = max(M_real_target.max(), M_imag_target.max())
    G_real_rram = map_weights_to_conductance_linear(M_real_target, g_min_siemens, g_max_siemens, effective_states, nonlinearity_param, max_target_val)
    G_imag_rram = map_weights_to_conductance_linear(M_imag_target, g_min_siemens, g_max_siemens, effective_states, nonlinearity_param, max_target_val)

    print("Mapping complete.\n")


    V_in = input_vector.reshape((N, 1))


    I_real_out_raw = G_real_rram @ V_in
    I_imag_out_raw = G_imag_rram @ V_in


    noise_scale = read_noise_std_scaled * (g_max_siemens - g_min_siemens)
    I_real_out_raw += np.random.normal(0, noise_scale, I_real_out_raw.shape)
    I_imag_out_raw += np.random.normal(0, noise_scale, I_imag_out_raw.shape)


    gain = max_target_val / (g_max_siemens - g_min_siemens)
    input_sum = np.sum(V_in)
    g_min_bias = g_min_siemens * input_sum
    offset_bias = offset_D * input_sum

    def post_process(I_out_raw, gain, g_min_bias, offset_bias):
        I_removed_bias = I_out_raw - g_min_bias
        W_scaled = I_removed_bias * gain
        W_final = W_scaled - offset_bias
        return W_final.flatten()

    output_real_rram = post_process(I_real_out_raw, gain, g_min_bias, offset_bias)
    output_imag_rram = post_process(I_imag_out_raw, gain, g_min_bias, offset_bias)


    output_ideal = WN @ V_in
    output_real_ideal = np.real(output_ideal).flatten()
    output_imag_ideal = np.imag(output_ideal).flatten()


    freq_axis = np.fft.fftfreq(N, d=Ts) / 1e9
    freq_shifted = np.fft.fftshift(freq_axis)

    mag_rram = np.sqrt(output_real_rram**2 + output_imag_rram**2)
    mag_ideal = np.sqrt(output_real_ideal**2 + output_imag_ideal**2)

    mag_rram_shifted = np.fft.fftshift(mag_rram)
    mag_ideal_shifted = np.fft.fftshift(mag_ideal)

    real_rram_shifted = np.fft.fftshift(output_real_rram)
    real_ideal_shifted = np.fft.fftshift(output_real_ideal)

    imag_rram_shifted = np.fft.fftshift(output_imag_rram)
    imag_ideal_shifted = np.fft.fftshift(output_imag_ideal)


    results_df = pd.DataFrame({
        'Frequency_GHz': freq_shifted,
        'Magnitude_Ideal': mag_ideal_shifted,
        'Magnitude_RRAM': mag_rram_shifted,
        'Real_Ideal': real_ideal_shifted,
        'Real_RRAM': real_rram_shifted,
        'Imag_Ideal': imag_ideal_shifted,
        'Imag_RRAM': imag_rram_shifted
    })

    save_filename = 'rram_dft_simulation_results.csv'
    results_df.to_csv(save_filename, index=False)
    print(f"--- Simulation Data Saved to '{save_filename}' ---")


    plt.figure(figsize=(12, 6))
    plt.stem(freq_shifted, mag_ideal_shifted, linefmt='b-', markerfmt='bo', basefmt=" ", label='Ideal DFT')
    plt.stem(freq_shifted, mag_rram_shifted, linefmt='r:', markerfmt='rx', basefmt=" ", label='RRAM-based DFT')
    plt.title(f'DFT Magnitude Spectrum: Ideal vs. RRAM ({fs/1e9:.0f}GHz Sampling)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude')
    plt.legend(); plt.grid(True); plt.show()


    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].stem(freq_shifted, real_ideal_shifted, linefmt='b-', markerfmt='bo', basefmt=" ", label='Ideal')
    axs[0].stem(freq_shifted, real_rram_shifted, linefmt='r:', markerfmt='rx', basefmt=" ", label='RRAM')
    axs[0].set_title('Real Part of DFT Output (Noise reflected)'); axs[0].set_ylabel('Amplitude')
    axs[0].legend(); axs[0].grid(True)

    axs[1].stem(freq_shifted, imag_ideal_shifted, linefmt='b-', markerfmt='bo', basefmt=" ", label='Ideal')
    axs[1].stem(freq_shifted, imag_rram_shifted, linefmt='r:', markerfmt='rx', basefmt=" ", label='RRAM')
    axs[1].set_title('Imaginary Part of DFT Output'); axs[1].set_xlabel('Frequency (GHz)'); axs[1].set_ylabel('Amplitude')
    axs[1].legend(); axs[1].grid(True)
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(M_real_ideal, cmap='viridis', aspect='auto')
    plt.colorbar(label='Ideal Weight')
    plt.title('Ideal DFT Matrix (Real Part)')

    plt.subplot(1, 2, 2)
    plt.imshow(G_real_rram * 1e3, cmap='viridis', aspect='auto')
    plt.colorbar(label='Physical Conductance (mS)')
    plt.title('Physical RRAM Matrix (Real Part, with Offset)')
    plt.tight_layout()
    plt.show()

    mse = np.mean(np.abs(mag_rram - mag_ideal)**2)
    print(f"Mean Squared Error (Magnitude): {mse:.4f}")

if __name__ == '__main__':
    main()
