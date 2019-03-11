import numpy as np
import math

from src.timeseries.TimeSeries import calcIncreamentalMeanStddev
from src.timeseries.TimeSeries import TimeSeries

import pyximport; pyximport.install(setup_args={'include_dirs':np.get_include()}, reload_support=True)

from src.transformation import cMFT

class MFT:

    def __init__(self, windowSize, normMean, lowerBounding, MUSE_Bool = False):
        self.windowSize = windowSize
        self.MUSE = MUSE_Bool

        self.startOffset = 2 if normMean else 0
        self.norm = 1.0 / np.sqrt(windowSize) if lowerBounding else 1.0


    def transform(self, series, wordlength):
        FFT_series = np.fft.fft(series)
        data_new = []
        windowSize = len(series)

        for i in range(int(math.ceil(len(series) / 2))):
            data_new.append(FFT_series[i].real)
            data_new.append(FFT_series[i].imag)
        data_new[1] = 0.0
        data_new = data_new[:self.windowSize]

        length = min([windowSize - self.startOffset, wordlength])
        copy = data_new[(self.startOffset):(length + self.startOffset)]
        while len(copy) != wordlength:
            copy.append(0)

        sign = 1
        for i in range(len(copy)):
            copy[i] *= self.norm * sign
            sign *= -1

        return copy


    def transformWindowing(self, series_full, wordLength):

        return cMFT.transformWindowing(self, series_full, wordLength)

    def normalizeFT(self, copy, std):

        return cMFT.normalizeFT(self, copy.data ,copy.NORM_CHECK , std)


