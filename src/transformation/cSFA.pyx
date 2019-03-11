import numpy as np

cimport numpy as np

def quantization(np.ndarray SFA_bins, list one_approx):
    cdef list word = [0 for _ in range(len(one_approx))]

    cdef int i = 0
    cdef int c
    cdef float v

    cdef list dims = [d for d in range(SFA_bins.shape[1])]

    for v in one_approx:
        c = 0
        for C in dims:
            if v < SFA_bins[i,c]:
                break
            else:
                c += 1
        word[i] = c-1
        i += 1

    return word