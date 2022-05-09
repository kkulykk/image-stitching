"""
Feature matching module
"""

import numpy as np

ratio_treshold = 0.6


def brute_force_matcher(features_query: list, descriptors_query: list, features_train: list, descriptors_train: list) -> tuple:
    result = []
    local_minimum = []
    for i in range(len(descriptors_query)):
        for k in range(len(descriptors_train)):
            dist = np.linalg.norm(descriptors_query[i]-descriptors_train[k])
            local_minimum.append((dist, k))
        local_minimum.sort
        if (local_minimum[0][1] / local_minimum[1][1]) < ratio_treshold and len(result) < 100:
            result.append(
                (features_query[i], features_train[local_minimum[0][1]]))
        local_minimum = []

    return result
