import numpy as np
from scipy.fftpack import fft

    
class CVD:

    def __call__(self, image):
        cvd_image =  np.abs(fft(image, axis=0),dtype=np.float32).transpose(1, 0)
        return cvd_image