import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from tqdm import tqdm

def plot_lfp(eeg_data, ts, start, duration, channels, color='k'):
    """
    Plot EEG data vertically.

    Parameters
    ----------
    eeg_data : ndarray
        EEG data.
    ts : ndarray
        Timestamps.
    start : float
        Time in seconds to start plotting.
    duration : float
        Duration in seconds to plot from start.
    channels : list
        List of channels to plot.
    color : str
        Color of the lines.
    """

    # find the start and end indices in timestamps array
    start_idx = np.argmin(np.abs(ts - start))
    end_idx = np.argmin(np.abs(ts - (start + duration)))

    # find maximum voltage to space the channels in the plot
    spacing = np.amax(abs(eeg_data[channels, start_idx:end_idx])) * 2
    
    plt.plot(ts[start_idx:end_idx], eeg_data[channels, start_idx:end_idx].T + spacing * np.arange(len(channels) - 1, -1, -1))

    plt.yticks(spacing * np.arange(len(channels) - 1, -1, -1), [f"Channel {i}" for i in channels])
    plt.xlabel("Time (s)")
    plt.ylabel("Channels")
    plt.savefig(f"plots/lfp.png")
    # plt.show()



def erp(signal, stimuli_idx, channels, lower_bound, upper_bound, fs=1000, num_stimuli=0, bad_threshold = 2000, plot=True):
    """
    Calculate the evoked response potential.

    Parameters
    ----------
    signal : ndarray
        Signal to calculate the evoked response potential from.
    stimuli_idx : ndarray
        Indices of the stimuli.
    channels : list
        Channels to calculate the evoked response potential from.
    lower_bound : float
        Lower bound of the time window to calculate the evoked response potential from.
    upper_bound : float
        Upper bound of the time window to calculate the evoked response potential from.
    fs : float
        Sampling frequency.
    num_stimuli : int
        Number of stimuli to calculate the evoked response potential from, if 0, use all stimuli.
    bad_threshold : float
        Threshold for bad channels who will be excluded in the final result.
    plot : bool
        Whether to plot the evoked response potential.

    Returns
    -------
    ndarray
        Evoked response potential.
    list
        Number of good stimuli per channel.
    """

    if num_stimuli > 0:
        stimuli_idx = stimuli_idx[:num_stimuli]
    else:
        num_stimuli = len(stimuli_idx)
    
    start_offset = int(lower_bound * fs)
    end_offset = int(upper_bound * fs)
    max_idx = signal.shape[1]

    mean_response = np.zeros((len(channels), abs(start_offset) + end_offset))
    good_stims = []

    for i, channel in enumerate(channels):
        stims = num_stimuli
        for stim in stimuli_idx:
            if stim + start_offset > 0 and stim + end_offset < max_idx:
                response = signal[channel, stim + start_offset:stim + end_offset]
                if np.any(abs(response) > bad_threshold):
                    stims -= 1
                    continue
                else:
                    mean_response[i] += response
        mean_response[i] /= stims
        mean_response[i] = np.nan_to_num(mean_response[i])
        good_stims.append(stims)

    if plot:
        y_max = np.max(mean_response, axis=0)
        y_min = np.min(mean_response, axis=0)
        y_mean = np.mean(mean_response, axis=0)
        
        plt.fill_between(np.linspace(lower_bound, upper_bound, abs(start_offset) + end_offset), y_max, y_min, alpha=0.7)
        plt.plot(np.linspace(lower_bound, upper_bound, abs(start_offset) + end_offset), y_mean)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (uV)")
        plt.title(f"Evoked response potential for {channels}")
        plt.show()

    return mean_response, good_stims

def get_channel_idx(channel_names, channel, dead_channels={}):
    """
    Returns all indices of channels with channel_name.

    Parameters
    ----------
    channel_names : str
        list of channel names.
    channel : str
        Channel to find.
    dead_channels : set
        Dead channels.

    Returns
    -------
    list
        Indices of the channels.
    """

    return [i for i, name in enumerate(channel_names) if name == channel and i not in dead_channels]

def get_brain_region(channel_names, brain_region):
    """
    Returns all occurences of channel names with that brain_region.

    Parameters
    ----------
    channel_names : str
        list of channel names.
    brain_region : str
        Brain region to find.

    Returns
    -------
    list
        names of the channels.
    """

    return [name for name in channel_names if name.startswith(brain_region)]

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    """
    Butterworth bandstop filter.
    credits: https://dsp.stackexchange.com/questions/49433/bandstop-filter

    Parameters
    ----------
    data : ndarray
        Data to filter.
    lowcut : float
        Lowcut frequency.
    highcut : float
        Highcut frequency.
    fs : float
        Sampling frequency.
    order : int
        Order of the filter.
    
    Returns
    -------
    ndarray 
        Filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = sig.butter(order, [low, high], btype='bandstop')
    y = sig.lfilter(i, u, data)
    return y



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y

def find_stimuli(stimuli_channel, threshold, fs, min_spacing):
    """
    Find stimuli in the stimuli channel.

    Parameters
    ----------
    stimuli_channel : ndarray
        Stimuli channel.
    threshold : float
        Threshold for the stimuli channel.
    fs : float
        Sampling frequency.
    min_spacing : float
        Minimum spacing between stimuli in seconds.

    Returns
    -------
    ndarray
        Indices of the stimuli.
    """

    # find all stimuli
    stimuli_idx = np.where(stimuli_channel > threshold)[0]

    # find the start of each stimulus
    stimuli_starts = []
    for idx in stimuli_idx:
        if len(stimuli_starts) == 0:
            stimuli_starts.append(idx)
        elif idx - stimuli_starts[-1] > min_spacing * fs:
            stimuli_starts.append(idx)
        
    return stimuli_starts

def car(signal, tetrodes, channel_names):
    """
    Calculate the common average reference of a given tetrode.

    Parameters
    ----------
    signal : ndarray
        Signal to calculate the common average reference from.
    tetrodes : list
        List of unique tetrodes
    channel_names : list
        List of all channel names.

    Returns
    -------
    ndarray
        Common average reference for a tetrode.
    """

    tetrode_set = set(tetrodes)
    car_signal = np.zeros(signal.shape)

    for trode in tetrodes:
        channels = get_channel_idx(channel_names, trode)
        diff_channels = set(range(128)) - set(channels)
        diff_mean = np.mean(signal[list(diff_channels)], axis=0)
        for channel in channels:
            car_signal[channel] = signal[channel] - diff_mean

    return car_signal


    # car_TT = np.empty((len(TT_channels), signal.shape[1]))
    # channel_set = set(TT_channels)

    # for idx, ch in enumerate(channel_set):
    #     ch_diff = list(channel_set.difference(set([ch]))) # get all channels except ch
    #     car_TT[idx] = signal[ch] - np.mean(signal[ch_diff], axis=0)
         
    # return car_TT






