import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
 
df = pd.read_csv('eeg_recording_raw.csv')
 
print("Columns:", df.columns)
# Assume last column is 'timestamp' and others are channels
timestamps = df['timestamp'].values
eeg_data = df.drop(columns=['timestamp']).values  # shape: (n_samples, n_channels)
 
#print("EEG shape:", eeg_data.shape)

fs_est = 1.0 / np.mean(np.diff(timestamps))
#print("Estimated fs:", fs_est)

fs = 250  # known sampling rate

#6.1 Simple MNE filter
import mne
 
# MNE expects (n_channels, n_samples)
eeg_data_T = eeg_data.T
 
info = mne.create_info(
    ch_names=[f"ch{i}" for i in range(eeg_data.shape[1])],
    sfreq=fs,
    ch_types='eeg'
)
raw = mne.io.RawArray(eeg_data_T, info)
 
# Apply bandpass 1–40 Hz
raw_filtered = raw.copy().filter(l_freq=1., h_freq=40., fir_design='firwin')
 
filtered_data = raw_filtered.get_data().T  # back to (n_samples, n_channels)
print("Filtered shape:", filtered_data.shape)

 
t = timestamps - timestamps[0]
ch_idx = 0
 
plt.figure(figsize=(10, 4))
plt.plot(t, eeg_data[:, ch_idx], label='Raw', alpha=0.5)
plt.plot(t, filtered_data[:, ch_idx], label='Filtered (1–40 Hz)', alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.title(f"Channel {ch_idx}: raw vs filtered")
plt.tight_layout()
plt.show()


#7. Step 3 – Compute Power Spectral Density (PSD)
#We’ll use Welch’s method (via SciPy or MNE).

#7.1 Using SciPy (low-level, good teaching moment)
from scipy.signal import welch
 
# Pick one channel to start with
ch_idx = 0
segment = filtered_data[:, ch_idx]
 
# Welch PSD
f, psd = welch(
    segment,
    fs=fs,
    nperseg=fs*2  # 2-second windows
)
 
print("Frequency bins shape:", f.shape)
print("PSD shape:", psd.shape)
#7.2 Plot the PSD
plt.figure(figsize=(8, 4))
plt.semilogy(f, psd)  # semilog-y to emphasize differences
plt.xlim(0, 40)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (µV²/Hz)")
plt.title(f"PSD for Channel {ch_idx}")
plt.tight_layout()
plt.show()