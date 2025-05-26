import numpy as np


def rayleigh_channel(signal, snr_db):
    """
    Simulate the Rayleigh fading channel by adding multipath
    fading and Gaussian noise to the input signal.

    Parameters:
        signal (numpy.ndarray): Input signal.
        snr_db (float): Signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: Signal with added Rayleigh fading and Gaussian noise.
    """
    signal_power = np.sum(np.abs(signal)**2) / len(signal)
    noise_power = signal_power / (10**(snr_db / 10))

    # Generate Rayleigh fading channel coefficients (complex Gaussian)
    fading_channel = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) / np.sqrt(2)

    # Generate additive white Gaussian noise
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

    # Apply Rayleigh fading and add noise
    received_signal = signal * fading_channel + noise

    return received_signal

