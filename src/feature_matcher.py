"""
Feature matching module
"""

from unittest import result
import numpy as np

ratio_treshold = 0.8


def knn(descriptor_query, descriptors_train: list, k: int = 2) -> list:
    """
    Return a list with k nearest neighbors for a given point using its' descriptors
    """
    neighbors = []
    for i in range(len(descriptors_train)):
        dist = np.linalg.norm(descriptor_query - descriptors_train[i])
        neighbors.append((dist, i))
    neighbors.sort()
    if neighbors[0][0] < (ratio_treshold * neighbors[1][0]):
        return neighbors[0][1]
    else:
        return -1


def feature_matcher(features_query: list, descriptors_query: list, features_train: list, descriptors_train: list, crossCheck: bool = False) -> list:
    """
    Return list of all matches between two images in form (keypoint_1, keypoint_2)
    which denotes a keypoint from query image and its representative from
    train image respectively
    """
    print("\nStart matching features\n")
    result1 = []
    result2 = []
    result = []
    for i in range(len(descriptors_query)):
        nearest_neighbor = knn(descriptors_query[i], descriptors_train)
        if nearest_neighbor != -1:
            result1.append(
                (features_train[nearest_neighbor], features_query[i]))
    if crossCheck:
        for i in range(len(descriptors_train)):
            nearest_neighbor = knn(descriptors_train[i], descriptors_query)
            if nearest_neighbor != -1:
                result2.append(
                    (features_query[nearest_neighbor], features_train[i]))

        for i in result1:
            if i in result2 or (i[1], i[0]) in result2:
                result.append(i)
    else:
        result = result1

    print("\nEnd matching features\n")
    return result
