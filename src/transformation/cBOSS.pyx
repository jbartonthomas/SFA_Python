def createWord(list numbers, int maxF, int bits):
    cdef int shortsPerLong = int(round(60 / bits))
    cdef int to = min([len(numbers), maxF])

    cdef int b = 0
    cdef int s = 0
    cdef int shiftOffset = 1
    cdef int i

    for i in range(s, (min(to, shortsPerLong + s))):
        shift = 1
        for j in range(bits):
            if (numbers[i] & shift) != 0:
                b |= shiftOffset
            shiftOffset <<= 1
            shift <<= 1

    cdef int limit = 2147483647
    cdef int total = 2147483647 + 2147483648
    while b > limit:
        b = b - total - 1
    return b