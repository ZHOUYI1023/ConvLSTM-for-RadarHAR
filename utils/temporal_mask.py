import numpy as np
import random

class TemporalMask:
    def __init__(self,
                 num_mask_max=1,
                 time_masked_ratio_max=0.50,
                 value=0):
        self.num_mask_max = num_mask_max
        self.time_masked_ratio_max = time_masked_ratio_max
        self.value = value

    def __call__(self, spec):
        spec = np.asarray(spec)
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
