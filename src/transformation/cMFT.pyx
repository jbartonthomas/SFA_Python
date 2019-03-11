from src.timeseries.TimeSeries import TimeSeries

import numpy as np

cimport numpy as np

import math


def calcIncreamentalMeanStddev(int windowLength, list series, list MEANS, list STDS):
    cdef float SUM = 0.
    cdef float squareSum = 0.

    cdef float rWindowLength = 1.0 / windowLength

    cdef float buf
    cdef int ww

    for ww in range(windowLength):
        SUM += series[ww]
        squareSum += series[ww]*series[ww]
    MEANS.append(SUM * rWindowLength)
    buf = squareSum*rWindowLength - MEANS[0]*MEANS[0]

    if buf > 0:
        STDS.append(np.sqrt(buf))
    else:
        STDS.append(0)

    for w in range(1,(len(series)-windowLength+1)):
        SUM += series[w+windowLength-1] - series[w-1]
        MEANS.append(SUM * rWindowLength)

        squareSum += series[w+windowLength-1]*series[w+windowLength-1] - series[w-1]*series[w-1]
        buf = squareSum * rWindowLength - MEANS[w]*MEANS[w]
        if buf > 0:
            STDS.append(np.sqrt(buf))
        else:
            STDS.append(0)

    return MEANS, STDS


def normalizeFT(self, list copy_data, bint NORM_CHECK, int std):
    cdef float normalisingFactor = 1. / std if (NORM_CHECK) & (std > 0) else 1.
    normalisingFactor *= self.norm

    cdef int sign = 1
    cdef int i

    for i in range(len(copy_data)):
        copy_data[i] *= sign * normalisingFactor
        sign *= -1

    return copy_data



def transformWindowing(self, series_full, wordLength):
    
    cdef list series = series_full.data
    cdef int WORDLENGTH = max(self.windowSize, wordLength + self.startOffset) if self.MUSE else min(self.windowSize, wordLength + self.startOffset)
    cdef int t
    cdef int k
    cdef int j
    cdef float real1
    cdef float imag1
    cdef float real
    cdef float imag

    WORDLENGTH = WORDLENGTH + WORDLENGTH % 2
    cdef list phis = [0. for i in range(WORDLENGTH)]

    cdef float uHalve

    for u in range(0, WORDLENGTH, 2):
        uHalve = -u / 2
        phis[u] = math.cos(2 * math.pi * uHalve / self.windowSize)
        phis[u+1] = -math.sin(2 * math.pi * uHalve / self.windowSize)

    cdef int final = max(1, len(series) - self.windowSize + 1)

    self.MEANS = []
    self.STDS = []

    self.MEANS, self.STDS = calcIncreamentalMeanStddev(self.windowSize, series, self.MEANS, self.STDS)

    cdef list transformed = []

    cdef list data = series
    cdef list mftData_FFT = [0. for _ in range(WORDLENGTH)]
    cdef list copy_value
    cdef list copy

    for t in range(final):
        if t > 0:
            k = 0
            while k < WORDLENGTH:
                real1 = mftData_FFT[k] + data[t + self.windowSize-1] - data[t-1]
                imag1 = mftData_FFT[k + 1]

                real = (real1 * phis[k]) - (imag1 * phis[k + 1])
                imag = (real1 * phis[k + 1]) + (imag1 * phis[k])
                mftData_FFT[k] = real
                mftData_FFT[k + 1] = imag
                k += 2
        else:
            mftData_fft = np.fft.fft(data[:self.windowSize])
            mftData_FFT = [0. for _ in range(WORDLENGTH)]

            i = 0
            for j in range(min(self.windowSize, WORDLENGTH)):
                if j % 2 == 0:
                    mftData_FFT[j] = mftData_fft[i].real
                else:
                    mftData_FFT[j] = mftData_fft[i].imag
                    i += 1
            mftData_FFT[1] = 0.


        copy = [0. for i in range(wordLength)]
        copy_value = mftData_FFT[(self.startOffset):(self.startOffset + wordLength)]
        copy[:len(copy_value)] = copy_value

        copy_ts = TimeSeries(copy, series_full.label, series_full.NORM_CHECK)
        copy_ts = self.normalizeFT(copy_ts, self.STDS[t])
        transformed.append(copy_ts)

    return transformed