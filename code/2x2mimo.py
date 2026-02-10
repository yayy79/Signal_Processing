#4QAM 2x2 MIMO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

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

def complex_to_real_matrix_expansion(W_complex):
    Rows, Cols = W_complex.shape
    W_real_expanded = np.zeros((2*Rows, 2*Cols))
    W_real_expanded[0:Rows, 0:Cols] = np.real(W_complex)
    W_real_expanded[0:Rows, Cols:]  = -np.imag(W_complex)
    W_real_expanded[Rows:, 0:Cols]  = np.imag(W_complex)
    W_real_expanded[Rows:, Cols:]   = np.real(W_complex)
    return W_real_expanded

def complex_to_real_vector_expansion(y_complex):
    return np.concatenate((np.real(y_complex), np.imag(y_complex)))

def real_to_complex_vector_restoration(x_real_expanded):
    N = len(x_real_expanded) // 2
    return x_real_expanded[:N] + 1j * x_real_expanded[N:]

def gray_to_binary(n):
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return n

def generate_gray_qam_constellation(M):
    bits_per_symbol = int(np.log2(M))
    constellation = np.zeros(M, dtype=complex)

    if M == 4:
        for i in range(M):
            i_bit = (i >> 1) & 1
            q_bit = i & 1
            i_idx = gray_to_binary(i_bit)
            q_idx = gray_to_binary(q_bit)
            I_val = 2 * i_idx - 1
            Q_val = 2 * q_idx - 1
            constellation[i] = I_val + 1j * Q_val

    elif M == 16:
        k_half = 2
        mask_half = 0x3
        for i in range(M):
            i_bits_gray = (i >> k_half) & mask_half
            q_bits_gray = i & mask_half
            i_idx = gray_to_binary(i_bits_gray)
            q_idx = gray_to_binary(q_bits_gray)
            I_val = 2 * i_idx - 3
            Q_val = 2 * q_idx - 3
            constellation[i] = I_val + 1j * Q_val

    return constellation / np.sqrt(np.mean(np.abs(constellation)**2))

def bits_to_indices(bits, bits_per_symbol):
    n_symbols = len(bits) // bits_per_symbol
    bits_reshaped = bits[:n_symbols*bits_per_symbol].reshape(n_symbols, bits_per_symbol)
    powers = 1 << np.arange(bits_per_symbol)[::-1]
    return bits_reshaped @ powers

def indices_to_bits(indices, bits_per_symbol):
    n_symbols = len(indices)
    bits = np.zeros((n_symbols, bits_per_symbol), dtype=int)
    for i in range(bits_per_symbol):
        shift = bits_per_symbol - 1 - i
        bits[:, i] = (indices >> shift) & 1
    return bits.flatten()

def calculate_evm(x_ideal, x_rx):
    Nt, num_symbols = x_ideal.shape
    evm_per_antenna = []

    error_vector = x_rx - x_ideal

    for i in range(Nt):
        p_avg = np.mean(np.abs(x_ideal[i, :])**2)

        mse = np.mean(np.abs(error_vector[i, :])**2)

        evm_rms = np.sqrt(mse / p_avg) * 100
        evm_per_antenna.append(evm_rms)

    overall_evm_percent = np.mean(evm_per_antenna)

    safe_evm = max(overall_evm_percent, 1e-10)

    return overall_evm_percent, safe_evm, evm_per_antenna

def run_mimo_simulation(Nt, Nr, M_QAM, SNR_dB, num_symbols, rram_params):
    g_min, g_max, num_states, nonlinearity, read_noise_std = rram_params
    bits_per_symbol = int(np.log2(M_QAM))
    total_bits_per_stream = num_symbols * bits_per_symbol

    constellation = generate_gray_qam_constellation(M_QAM)
    tx_bits_streams = []
    tx_indices_streams = []
    x_tx = np.zeros((Nt, num_symbols), dtype=complex)

    for i in range(Nt):
        bits = np.random.randint(0, 2, total_bits_per_stream)
        indices = bits_to_indices(bits, bits_per_symbol)
        tx_bits_streams.append(bits)
        tx_indices_streams.append(indices)
        x_tx[i, :] = constellation[indices]

    H = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
    Es = 1.0
    N0 = Es / (10**(SNR_dB/10))
    noise = (np.random.randn(Nr, num_symbols) + 1j * np.random.randn(Nr, num_symbols)) * np.sqrt(N0/2)
    y_rx = H @ x_tx + noise

    H_herm = H.conj().T
    W_mmse_complex = np.linalg.inv(H_herm @ H + N0 * np.eye(Nt)) @ H_herm
    W_mmse_real = complex_to_real_matrix_expansion(W_mmse_complex)

    min_val = np.min(W_mmse_real)
    offset_D = -min_val
    W_target = W_mmse_real + offset_D
    max_target_val = np.max(W_target)
    G_rram = map_weights_to_conductance_linear(W_target, g_min, g_max, num_states, nonlinearity, max_target_val)

    y_rx_real = complex_to_real_vector_expansion(y_rx)
    gain_slope = (g_max - g_min) / max_target_val if max_target_val > 0 else 1
    I_out_raw = G_rram @ y_rx_real

    current_noise_scale = np.max(np.abs(I_out_raw)) * read_noise_std
    I_out_noisy = I_out_raw + np.random.normal(0, current_noise_scale, I_out_raw.shape)

    y_sum = np.sum(y_rx_real, axis=0)
    I_removed_bias = I_out_noisy - (g_min * y_sum)
    W_scaled_out = I_removed_bias / gain_slope
    x_hat_real_rram = W_scaled_out - (offset_D * y_sum)
    x_hat_rram = real_to_complex_vector_restoration(x_hat_real_rram)
    x_hat_ideal = x_tx

    def detect_and_count_errors(signal_complex, tx_indices, tx_bits, constellation, bps):
        total_bit_err, total_sym_err = 0, 0
        per_ant = []
        for ant in range(signal_complex.shape[0]):
            points = signal_complex[ant, :]
            dists = np.abs(points[:, None] - constellation[None, :])
            det_idx = np.argmin(dists, axis=1)

            true_idx = tx_indices[ant]
            sym_err = np.sum(det_idx != true_idx)

            det_bits = indices_to_bits(det_idx, bps)
            true_bits = tx_bits[ant]
            bit_err = np.sum(det_bits != true_bits)

            total_bit_err += bit_err
            total_sym_err += sym_err
            per_ant.append({'ber': bit_err/len(true_bits), 'ser': sym_err/len(true_idx)})
        return total_bit_err, total_sym_err, per_ant

    bit_err_rram, sym_err_rram, stats_rram = detect_and_count_errors(
        x_hat_rram, tx_indices_streams, tx_bits_streams, constellation, bits_per_symbol)

    total_bits = Nt * num_symbols * bits_per_symbol
    total_syms = Nt * num_symbols

    return bit_err_rram/total_bits, sym_err_rram/total_syms, x_hat_ideal, x_hat_rram, constellation, stats_rram

def main():
    Nt, Nr = 2, 2
    M_QAM = 4
    SNR_dB = 12
    num_symbols = 200000
    np.random.seed(20)
    g_min, g_max, num_states, nonlinearity, read_noise_std = 1e-3, 8e-3, 76, 0.38, 0.0025
    rram_params = (g_min, g_max, num_states, nonlinearity, read_noise_std)

    print(f"--- {Nt}x{Nr} MIMO MMSE Detection (4-QAM Mode) ---")
    print(f"RRAM: {num_states} States, Noise={read_noise_std*100}%")

    ber, ser, x_ideal, x_rram, constellation, stats = run_mimo_simulation(
        Nt, Nr, M_QAM, SNR_dB, num_symbols, rram_params
    )

    evm_pct, evm_db, evm_per_ant = calculate_evm(x_ideal, x_rram)

    print(f"\n=== Overall Results (SNR={SNR_dB}dB) ===")
    print(f"RRAM Average BER: {ber:.6f}")
    print(f"RRAM Average SER: {ser:.6f}")

    print("\n--- Saving Summary Results to CSV ---")

    summary_list = []
    for i, stat in enumerate(stats):
        summary_list.append({
            'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Antenna': i + 1,
            'SNR_dB': SNR_dB,
            'Total_BER': ber,
            'Total_SER': ser,
            'Total_EVM_Pct': evm_pct,
            'Total_EVM_dB': evm_db,
            'Antenna_BER': stat['ber'],
            'Antenna_SER': stat['ser'],
            'Antenna_EVM_Pct': evm_per_ant[i],
            'M_QAM': M_QAM,
            'num_states': num_states,
            'nonlinearity': nonlinearity,
            'read_noise_std': read_noise_std
        })

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv('mimo_4qam_rram_summary.csv', index=False, encoding='utf-8-sig')
    print("1. 'mimo_4qam_rram_summary.csv' saved.")

    print("--- Saving Signal IQ Data (All Symbols) ---")

    signal_data = {}
    for i in range(Nt):
        signal_data[f'Ant{i+1}_Rx_I'] = np.real(x_rram[i, :])
        signal_data[f'Ant{i+1}_Rx_Q'] = np.imag(x_rram[i, :])
        signal_data[f'Ant{i+1}_Tx_I'] = np.real(x_ideal[i, :])
        signal_data[f'Ant{i+1}_Tx_Q'] = np.imag(x_ideal[i, :])

    df_signals = pd.DataFrame(signal_data)
    df_signals.to_csv('mimo_4qam_iq_signals.csv', index=False)
    print("2. 'mimo_4qam_iq_signals.csv' saved.")

    plt.figure(figsize=(14, 7))
    plt.suptitle(f"{Nt}x{Nr} MIMO (4-QAM, SNR={SNR_dB}dB)\nEVM: {evm_pct:.2f}% | Total Sym: {num_symbols}", fontsize=14)

    for i in range(Nt):
        plt.subplot(1, Nt, i+1)
        plt.title(f"Ant {i+1} (EVM: {evm_per_ant[i]:.2f}%)\nBER={stats[i]['ber']:.5f}")
        plt.scatter(np.real(x_rram[i, :]), np.imag(x_rram[i, :]),
                    c='red', s=0.5, alpha=0.05, label='RRAM Rx', edgecolors='none')
        plt.xlabel('I'); plt.ylabel('Q'); plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim([-2.5, 2.5]); plt.ylim([-2.5, 2.5]); plt.axis('equal')
        if i == 0:
            leg = plt.legend(loc='upper right')
            for lh in leg.legend_handles:
                lh.set_alpha(1); lh._sizes = [30]

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
