def calculate_frequencies(time, signals):
    """
    Calculates the frequency of each signal in the given data.

    This function takes a 1D numpy array of time values and a 2D numpy array of signals as input, where each column of
    the signals array represents a different signal, and returns a list containing the frequency of each signal.

    Parameters
    ----------
    time : array_like
        The time values of the input data.
    signals : array_like
        The input signals, where each column represents a different signal.

    Returns
    -------
    list
        A list containing the frequency of each signal.
    """

    # Calculate FFT of each signal
    fft_values = np.fft.fft(signals, axis=0)

    # Calculate power spectrum of each signal
    L = signals.shape[0]
    P2 = np.abs(fft_values / L)
    P1 = P2[: L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]

    # Ignore 0 Hz frequency component
    P1[0] = 0

    # Find index of peak frequency for each signal
    peak_indices = np.argmax(P1, axis=0)

    # Calculate frequency values
    Fs = 1 / np.mean(np.diff(time))  # Sampling frequency
    f_values = Fs * np.arange(L // 2 + 1) / L

    # Calculate frequency of each signal
    frequencies = f_values[peak_indices]

    return list(frequencies)


def rps_order_checker(rps_data: np.ndarray):
    print("_" * 60, "order checker", "_" * 60)
    freq_input_test = 1000
    time = np.linspace(25, 25.01, 100)  # rps_data[100000:100100, 0]
    sinP = np.sin(
        freq_input_test * 2 * np.pi * time
    )  # rps_data[100000:100100, 1]  # sinP
    cosP = -np.cos(
        freq_input_test * 2 * np.pi * time
    )  # rps_data[100000:100100, 2]  # cosP
    sinN = -np.sin(
        freq_input_test * 2 * np.pi * time
    )  # rps_data[100000:100100, 3]  # sinN
    cosN = np.cos(
        freq_input_test * 2 * np.pi * time
    )  # rps_data[100000:100100, 4]  # cosN
    cosN = normalise_signal(cosN)
    cosP = normalise_signal(cosP)
    sinN = normalise_signal(sinN)
    sinP = normalise_signal(sinP)
    signals = np.column_stack((sinP, sinN, cosP, cosN))

    fig12, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    ax1.plot(time, sinP)
    ax2.plot(time, cosP)
    ax3.plot(time, sinN)
    ax4.plot(time, cosN)
    ax5.plot(time, sinP, label="SinP")
    ax5.plot(time, cosP, label="SinN")
    ax5.plot(time, sinN, label="CosP")
    ax5.plot(time, cosN, label="CosN")
    plt.legend()

    frequencies = calculate_frequencies(time, signals)

    print(f"The frequencies of the signals are {frequencies}.")

    print("=" * 120, "\n")
    return 0


# %% Toplevel Runner
if __name__ == "__main__":
    rps_data_np_V = rps_prefilter(df_filepath_V, df_test_V, eol_test_id_V)
    # rps_zero_status = rps_signal_zero_checker(rps_data_np_V)
    # rps_short_status = rps_signal_5V_checker(rps_data_np_V)
    # rps_static_status = rps_signal_static_checker(rps_data_np_V)
    rps_order_status = rps_order_checker(rps_data_np_V)

    print("_" * 60, "Results", "_" * 60)
    # print(rps_zero_status)
    # print(rps_short_status)
    # print(f"Overall Results: {rps_static_status[0]}")
    # print(f"Average Status: {rps_static_status[1]}")
    # print(f"Differential Status: {rps_static_status[2]}")
    # print(f"Non Normal Times: {rps_static_status[3]}")
    # print(f"Differential RMS values: {rps_static_status[4]}")
    print("=" * 120, "\n")
    plt.show()
