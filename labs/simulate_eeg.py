import numpy as np
import matplotlib.pyplot as plt
 
fs = 250           # sampling rate
duration = 10      # seconds
t = np.arange(0, duration, 1/fs)
 
# Simulate alpha rhythm at 10 Hz + noise
alpha_freq = 10
alpha_signal = 50 * np.sin(2 * np.pi * alpha_freq * t)
noise = 20 * np.random.randn(len(t))
 
sim_channel = alpha_signal + noise
 
# Multiple channels: copy with slight variations
n_channels = 4
samples = np.vstack([
    sim_channel + np.random.randn(len(t))*5 for _ in range(n_channels)
]).T  # shape (n_samples, n_channels)
 
timestamps = t
 
print("Simulated data shape:", samples.shape)
 
plt.figure(figsize=(10, 4))
plt.plot(t, samples[:, 0])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.title("Simulated EEG Channel 0")
plt.tight_layout()
plt.show()