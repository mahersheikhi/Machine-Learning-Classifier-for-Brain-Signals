import mne
from mne.preprocessing import ICA, create_eog_epochs
import numpy as np

def clean_eeg(eeg_data, fs, line_freq=50.0, n_components=20, eog_ch_idx=0):
    """
    eeg_data: (n_samples, n_channels)
    fs: sampling rate
    line_freq: notch frequency (50 or 60 Hz)
    eog_ch_idx: index of channel closest to eyes (or None if unknown)
    """
    
 
    eeg_T = eeg_data.T
    n_channels = eeg_data.shape[1]
    ch_names = [f"ch{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=['eeg']*n_channels)
    raw = mne.io.RawArray(eeg_T, info)
 
    # 1. Bandpass 1–40 Hz
    raw_filtered = raw.copy().filter(1., 40., fir_design='firwin')
 
    # 2. Notch filter
    raw_filtered = raw_filtered.notch_filter(freqs=[line_freq])
 
    # 3. ICA
    ica = ICA(n_components=min(n_components, n_channels), random_state=97, max_iter='auto')
    ica.fit(raw_filtered)
 
    # 4. Find EOG components (if eog_ch_idx is provided)
    #if eog_ch_idx is not None:
     #   eog_epochs = create_eog_epochs(raw_filtered, ch_name=ch_names[eog_ch_idx])
      #  eog_inds, scores = ica.find_bads_eog(eog_epochs)
       # ica.exclude = eog_inds
    #else:
     #   ica.exclude = []  # no automatic exclusion
 
    raw_clean = raw_filtered.copy()
    ica.apply(raw_clean)
 
    cleaned_data = raw_clean.get_data().T  # back to (n_samples, n_channels)
    return cleaned_data, raw, raw_filtered, raw_clean, ica

#simulate eeg

fs = 250
duration = 180
t = np.arange(0, duration, 1/fs)
n_samples = len(t)
n_channels = 4
 
# Base: alpha-like 10 Hz + noise
base = 20 * np.sin(2*np.pi*10*t) + 5*np.random.randn(n_samples)
 
eeg_sim = np.vstack([base + np.random.randn(n_samples)*2 for _ in range(n_channels)]).T
 
# Add synthetic 'blinks' to channel 0
blink_times = [5, 10, 15]  # seconds
for bt in blink_times:
    idx = (t > bt) & (t < bt+0.2)
    eeg_sim[idx, 0] += 150  # big spike



cleaned_data, raw_orig, raw_filt, raw_clean_obj, ica_obj = clean_eeg(eeg_sim, fs, eog_ch_idx=0)
 
print("Cleaned data shape:", cleaned_data.shape)