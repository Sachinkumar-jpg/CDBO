import numpy as np


def awgn(signal, snr):
  """
  Adds AWGN noise to a signal
  Args:
    signal: The input signal.
    snr: The signal-to-noise ratio in dB.

  Returns:
    The noisy signal.
  """

  # Calculate the noise power
  noise_power = signal.var() / (10**(snr / 10))

  # Generate the noise
  noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

  # Add the noise to the signal
  noisy_signal = signal + noise

  return noisy_signal
