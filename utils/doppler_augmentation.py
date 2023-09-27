import numpy as np
import random
from scipy.ndimage import zoom


class DopplerMask:
    def __init__(self,
                 num_mask_max=2,
                 freq_masked_ratio_max=0.15,
                 value=0):
        self.num_mask_max = num_mask_max
        self.freq_masked_ratio_max = freq_masked_ratio_max
        self.value = value

    def __call__(self, spec):
        assert (len(spec.shape) == 2)
        freq_count = spec.shape[0]
        spec = spec.copy()
        mask_count = random.randint(1, self.num_mask_max)  # random number of mask
        for i in range(mask_count):
            freq_masked_ratio = random.uniform(0.0, self.freq_masked_ratio_max)
            freq_masked_count = int(freq_masked_ratio * freq_count)
            f0 = int(np.random.uniform(low=0.0, high=freq_count - freq_masked_count))
            spec[f0:f0 + freq_masked_count, :] = self.value
        return spec


class DopplerScale:
    def __init__(self, ratio, static_bandwidth=40):
        self.scale_ratio = ratio
        self.static_bandwidth = static_bandwidth

    def __call__(self, signal):
        lower_ind = int((signal.shape[0]-self.static_bandwidth)/2)
        upper_ind = int((signal.shape[0] + self.static_bandwidth) / 2)
        signal_lower = signal[0:lower_ind,:]
        signal_middle = signal[lower_ind:upper_ind,:]
        signal_upper = signal[upper_ind:,:]
        signal_upper = zoom(signal_upper, [self.scale_ratio,1] , mode='nearest')
        signal_lower = zoom(signal_lower, [self.scale_ratio,1] , mode='nearest')
        signal_new = np.concatenate((signal_lower,signal_middle, signal_upper), axis=0)
        half_offset_1 = int(abs((signal.shape[0] - signal_new.shape[0])) / 2)
        half_offset_2 = abs(signal.shape[0] - signal_new.shape[0]) - half_offset_1
        if self.scale_ratio > 1:
            signal = signal_new[half_offset_1:-half_offset_2,:]
        else:
            pad_width = ((half_offset_1, half_offset_2), (0, 0))
            signal = np.pad(signal_new, pad_width, 'constant', constant_values=signal.min())
        return signal


class DopplerShift:
    def __init__(self, ratio,static_bandwidth=40):
        self.shift_ratio = ratio
        self.static_bandwidth = static_bandwidth

    def __call__(self, signal):
        shift_step = int(abs(self.shift_ratio * signal.shape[0]))
        lower_ind = int((signal.shape[0] - self.static_bandwidth) / 2)
        upper_ind = int((signal.shape[0] + self.static_bandwidth) / 2)
        signal_lower = signal[0:lower_ind, :]
        signal_middle = signal[lower_ind:upper_ind, :]
        signal_upper = signal[upper_ind:, :]
        signal_new = np.concatenate((signal_lower,signal_upper),axis=0)
        if self.shift_ratio > 0:
            signal_new = signal_new[shift_step:,:]
            pad_width = ((0,shift_step), (0, 0))
            signal_new = np.pad(signal_new, pad_width, 'constant', constant_values=signal.min())
        else:
            signal_new = signal_new[:-shift_step, :]
            pad_width = ((shift_step,0), (0, 0))
            signal_new = np.pad(signal_new, pad_width, 'constant', constant_values=signal.min())
        signal = np.concatenate((signal_new[:int(signal_new.shape[0] / 2), :],
                                 signal_middle,
                                 signal_new[int(signal_new.shape[0] / 2):, :]), axis=0)
        return signal
