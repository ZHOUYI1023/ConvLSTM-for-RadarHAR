import numpy as np
import random
from scipy.ndimage import zoom


class TemporalDrop:
    def __init__(self, drop_ratio):
        self.drop_ratio = drop_ratio

    def __call__(self, signal):
        drop_length = int(self.drop_ratio * signal.shape[1])
        ind = random.sample(range(0, signal.shape[1]), drop_length)
        mask = np.ones(signal.shape[1], dtype=bool)
        mask[ind] = False
        return signal[:, mask]


class RandomTemporalCrop:
    def __init__(self, crop_ratio):
        self.crop_ratio = crop_ratio

    def __call__(self, signal):
        crop_length = int(self.crop_ratio * signal.shape[1])
        start = random.randint(0, signal.shape[1] - crop_length)
        return signal[:, start: start + crop_length]


class TemporalPad:
    def __init__(self, size, mode='random'):
        self.size = size
        mode_list = ['center', 'left', 'right']
        if mode == 'random':
            ind = random.randint(0, 2)
            mode = mode_list[ind]
        else:
            assert mode in mode_list
        self.mode = mode

    def __call__(self, signal):
        if signal.shape[1] < self.size:
            padding = self.size - signal.shape[1]
            if self.mode == 'center':
                offset = padding // 2
                pad_width = ((0, 0), (offset, padding - offset))

            elif self.mode == 'left':
                pad_width = ((0, 0), (padding, 0))
            elif self.mode == 'right':
                pad_width = ((0, 0), (0, padding))
        signal = np.pad(signal, pad_width,
                        'constant', constant_values=signal.min())
        return signal


class TemporalScale:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        if signal.shape[1] < self.size:
            zoom_ratio = [1, self.size / signal.shape[1]]
            signal = zoom(signal, zoom_ratio, mode='nearest')
        return signal


class TemporalMask:
    def __init__(self,
                 num_mask_max=2,
                 time_masked_ratio_max=0.20,
                 value=0):
        self.num_mask_max = num_mask_max
        self.time_masked_ratio_max = time_masked_ratio_max
        self.value = value

    def __call__(self, spec):
        assert (len(spec.shape) == 2)
        frame_count = spec.shape[1]
        spec = spec.copy()
        mask_count = random.randint(1, self.num_mask_max)  # random number of mask
        for i in range(mask_count):
            time_masked_ratio = random.uniform(0.0, self.time_masked_ratio_max)
            frame_masked_count = int(time_masked_ratio * frame_count)
            t0 = int(np.random.uniform(low=0.0, high=frame_count - frame_masked_count))
            spec[:, t0:t0 + frame_masked_count] = self.value
        return spec

