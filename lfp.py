import numpy as np
import matplotlib.pyplot as plt

from open_ephys.analysis import Session
import pyopenephys as oe

from utils import evoked_response_potential, butter_bandstop_filter, get_channel_idx, get_brain_region, find_stimuli, plot_lfp


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
ts = np.asarray(rec2.analog_signals[0].times)

### Downsamping and filtering
print("Downsampling...")
fs_target = 1000
ds_factor = Fs // fs_target
ds_length = signal.shape[1] // ds_factor

downsampled_signal = np.empty((signal.shape[0], ds_length))
downsampled_ts = np.empty(ds_length)

for idx in range(ds_length):
    downsampled_signal[:, idx] = signal[:, idx * ds_factor] 
    downsampled_ts[idx] = ts[idx * ds_factor]

plot_lfp(downsampled_signal, downsampled_ts, 3600, 1800, [139, 140, 141, 142, 143, 144, 145, 146, 147])
