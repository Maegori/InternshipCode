import scipy.signal as sig
import numpy as np
import time

from open_ephys.analysis import Session
import pyopenephys as oe

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

# dead_channels = {105, 106, 90, 91, 66, 67, 68, 69, 70, 36, 37, 38, 125, 126, 127, 122, 111, 97, 98, 101, 102, 84, 29, 30, 31}


# cannot load data via session (permission denied)
# samples = session.recordnodes[0].recordings[0].continuous[0].samples

data_file = oe.File(EXP_FOLDER)
experiments = data_file.experiments

experiment = experiments[0]
recordings = experiment.recordings

rec1 = recordings[0]

signal = rec1.analog_signals[0].signal
# ts = np.asarray(rec1.analog_signals[0].times)

fs_target = 1000
ds_factor = Fs // fs_target
ds_length = signal.shape[1] // ds_factor

t0 = time.time()
downsampled_signal_loop = np.empty((signal.shape[0], ds_length))
# downsampled_ts = np.empty(ds_length)

for idx in range(ds_length):
    downsampled_signal_loop[:, idx] = signal[:, idx * ds_factor] 
    # downsampled_ts[idx] = ts[idx * ds_factor]

t1 = time.time()
print(f"Loop time: {t1 - t0}")

t0 = time.time()
downsampled_signal_dec = sig.decimate(signal, ds_factor)
t1 = time.time()
print(f"Decimate time: {t1 - t0}")

# t0 = time.time()
# downsampled_signal_resample = sig.resample(signal, ds_length)
# t1 = time.time()
# print(f"Resample time: {t1 - t0}")


# t0 = time.time()
# downsampled_signal_respoly = sig.resample_poly(signal, up=1, down=ds_factor)
# t1 = time.time()
# print(f"Resample poly time: {t1 - t0}")

print("Difference between loop vs decimate:", np.sum(np.abs(downsampled_signal_loop - downsampled_signal_dec)))
# print("Difference between loop vs resample:", np.sum(np.abs(downsampled_signal_loop - downsampled_signal_resample)))
# print("Difference between loop vs resample poly:", np.sum(np.abs(downsampled_signal_loop - downsampled_signal_respoly)))

