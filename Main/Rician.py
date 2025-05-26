import numpy as np


def rician_channel(signal, snr_db, K):
    """
    Simulate the Rician fading channel by adding LOS and scattered components to the input signal.

    Parameters:
        signal (numpy.ndarray): Input signal.
        snr_db (float): Signal-to-noise ratio in dB.
        K (float): Rician factor representing the strength of the
        LOS component relative to the scattered component.

    Returns:
        numpy.ndarray: Signal with added Rician fading and Gaussian noise.
    """
    signal_power = np.sum(np.abs(signal)**2) / len(signal)
    noise_power = signal_power / (10**(snr_db / 10))

    # Generate LOS and scattered components
    LOS = np.sqrt(K / (K + 1)) * signal
    scattered = np.sqrt(1 / (K + 1)) * np.random.randn(*signal.shape)

    # Generate additive white Gaussian noise
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

    # Add LOS, scattered, and noise components
    received_signal = LOS + scattered + noise

    return received_signal
