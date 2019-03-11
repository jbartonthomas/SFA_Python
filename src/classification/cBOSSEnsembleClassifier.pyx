
def prediction(list bag_test, list bag_train, list label_test, list label_train, training_check):
    cdef list p_labels = ['Nan' for j in range(len(label_test))]
    cdef int p_correct = 0
    cdef int i
    cdef int minDistance
    cdef int noMatchDistance
    cdef int disance = 0

    cdef float buf

    for i in range(len(bag_test)):
        minDistance = 2147483647

        noMatchDistance = 0
        for key in bag_test[i].keys():
            noMatchDistance += bag_test[i][key] ** 2

        for j in range(len(bag_train)):
            if (bag_train[j] != bag_test[i]) | (training_check):
                distance = 0
                for key in bag_test[i].keys():
                    buf = bag_test[i][key] - bag_train[j][key] if key in bag_train[j].keys() else bag_test[i][key]
                    distance += buf ** 2

                    if distance >= minDistance:
                        continue

                if (distance != noMatchDistance) & (distance < minDistance):
                    minDistance = distance
                    p_labels[i] = label_train[j]

        if label_test[i] == p_labels[i]:
            p_correct += 1

    return p_correct, p_labels