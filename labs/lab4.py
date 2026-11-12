from lab3 import eeg_sim
import numpy as np
import mne


eeg_data = eeg_sim
eeg_sig = np.asarray(eeg_data)
 
#Create MNE Raw Object + Filter

fs = 250
ch_names = [f"ch{i}" for i in range(eeg_sig.shape[1])]
info = mne.create_info(ch_names, fs, ch_types='eeg')
 
raw = mne.io.RawArray(eeg_sig.T, info)
raw.filter(1., 30.)  # ERPs need low freq

#Step 5 – Epoch On Stimulus Events

#We want epochs: −200 ms to +800 ms around each event
#Baseline: −200 to 0 ms
event_times = np.arange(1.0, len(eeg_sig) / fs - 0.8, 1.5)
event_samples = np.round(event_times * fs).astype(int)
event_rng = np.random.default_rng(seed=42)
event_codes = np.where(event_rng.random(len(event_samples)) < 0.2, 2, 1)
events = np.column_stack((
    event_samples,
    np.zeros(len(event_samples), dtype=int),
    event_codes,
))

epochs = mne.Epochs(
    raw, 
    events=events.astype(int),
    event_id={'standard':1, 'target':2},
    tmin=-0.2,
    tmax=0.8,
    baseline=(-0.2, 0)
)

#4.6 Step 6 – Compute ERP Averages
evoked_standard = epochs['standard'].average()
evoked_target   = epochs['target'].average()
 
#evoked_standard.plot(titles="Standard ERP")
#evoked_target.plot(titles="Target ERP")
