import numpy as np
import matplotlib.pyplot as plt

from open_ephys.analysis import Session
import pyopenephys as oe

from utils import evoked_response_potential, butter_bandstop_filter, get_channel_idx, get_brain_region, find_stimuli


import numpy as np
import matplotlib.pyplot as plt

from open_ephys.analysis import Session
import pyopenephys as oe

from utils import evoked_response_potential, butter_bandstop_filter, get_channel_idx, get_brain_region, find_stimuli

### Data Loading

#load data
DATA_FOLDER = "/data/jpatriota/R14/3.Probe test/2021-09-23_00-16-10"
EXP_FOLDER = "/data/jpatriota/R14/3.Probe test/2021-09-23_00-16-10/Record Node 107/"
session = Session(DATA_FOLDER)

info = session.recordnodes[0].recordings[0].info['continuous'][0]
Fs = info['sample_rate']
channels = info['channels']
channel_names = [c['channel_name'] for c in channels]
unique_channel_names = np.unique(channel_names)
unique_data_channel_names = np.unique(channel_names[0:127])
brain_regions = ["A1T", "BLAT", "HPCT", "PFCT"]


data_file = oe.File(EXP_FOLDER)
experiments = data_file.experiments

experiment = experiments[0]
recordings = experiment.recordings

rec1 = recordings[0]
rec2 = recordings[1]

signal = rec2.analog_signals[0].signal

### Downsamping and filtering
print("Downsampling...")
fs_target = 1000
ds_factor = Fs // fs_target
ds_length = signal.shape[1] // ds_factor

downsampled_signal = np.empty((signal.shape[0], ds_length))

for idx in range(ds_length):
    downsampled_signal[:, idx] = signal[:, idx * ds_factor] 
print("Done!")
print("Filtering...")
filtered_signal = butter_bandstop_filter(downsampled_signal[:128], lowcut=48, highcut=51, fs=fs_target, order=5)
print("Done!")

CSmin_idx = find_stimuli(downsampled_signal[144], threshold=10000, fs=fs_target, min_spacing=4.5)
CSplus_idx = find_stimuli(downsampled_signal[145], threshold=10000, fs=fs_target, min_spacing=2)
CSnovel_idx = find_stimuli(downsampled_signal[140], threshold=10000, fs=fs_target, min_spacing=4.5)

print("Done!")
print("CS+:", len(CSplus_idx))
print("CS-:", len(CSmin_idx))
print("CSn:", len(CSnovel_idx))

lower_bound = -1
upper_bound = 4

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 10))
axes = axes.flatten()

trodes = get_brain_region(unique_channel_names, brain_regions[0])

for idx, trode in enumerate(trodes):
    active_channels = get_channel_idx(channel_names, trode)
    for ch in active_channels:
        response = evoked_response_potential(filtered_signal, CSplus_idx, [ch], lower_bound, upper_bound, num_stimuli=0, plot=False)[0]
        response = (response - np.mean(response)) / np.std(response)
        axes[idx].plot(response, label=f"{ch+1}")
    
    axes[idx].axvline(fs_target, color='k', linestyle='--')  
    axes[idx].set_xlabel("Time (s)")
    axes[idx].set_ylabel("Amplitude (uV)")
    axes[idx].legend()
    axes[idx].set_title(f"Asleep {trode} CS+ Z-scored ERP, -1s 2s of stim 2s of data, {len(CSplus_idx)} stimuli")

fig.tight_layout()
fig.savefig(f"plots/ERP_asleep_CSplus_{brain_regions[0]}_alter.png")


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 10))
axes = axes.flatten()

trodes = get_brain_region(unique_channel_names, brain_regions[0])

for idx, trode in enumerate(trodes):
    active_channels = get_channel_idx(channel_names, trode)
    for ch in active_channels:
        response = evoked_response_potential(filtered_signal, CSmin_idx, [ch],lower_bound, upper_bound, num_stimuli=0, plot=False)[0]
        response = (response - np.mean(response)) / np.std(response)
        axes[idx].plot(response, label=f"{ch+1}")
    
    axes[idx].axvline(fs_target, color='k', linestyle='--')  
    axes[idx].set_xlabel("Time (s)")
    axes[idx].set_ylabel("Amplitude (uV)")
    axes[idx].legend()
    axes[idx].set_title(f"Asleep {trode} CS- Z-scored ERP, -1s 2s of stim 2s of data, {len(CSmin_idx)} stimuli")

fig.tight_layout()
fig.savefig(f"plots/ERP_asleep_CSmin_{brain_regions[0]}_alter.png")



fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 10))
axes = axes.flatten()

trodes = get_brain_region(unique_channel_names, brain_regions[0])

for idx, trode in enumerate(trodes):
    active_channels = get_channel_idx(channel_names, trode)
    for ch in active_channels:
        response = evoked_response_potential(filtered_signal, CSnovel_idx, [ch],lower_bound, upper_bound, num_stimuli=0, plot=False)[0]
        response = (response - np.mean(response)) / np.std(response)
        axes[idx].plot(response, label=f"{ch+1}")
    
    axes[idx].axvline(fs_target, color='k', linestyle='--')   
    axes[idx].set_xlabel("Time (s)")
    axes[idx].set_ylabel("Amplitude (uV)")
    axes[idx].legend()
    axes[idx].set_title(f"Asleep {trode} CS novel Z-scored ERP, -1s 2s of stim 2s of data, {len(CSnovel_idx)} stimuli")

fig.tight_layout()
fig.savefig(f"plots/ERP_asleep_CSnovel_{brain_regions[0]}_alter.png")

