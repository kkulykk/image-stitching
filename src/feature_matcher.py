"""
Feature matching module
"""

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
        neighbors.sort
    return neighbors[:k]


def ratio_test(neighbors: list) -> bool:
    """
    Return true if the ratio between two nearest neighbors is below given threshold
    """
    return (neighbors[1][0] / neighbors[0][0]) < ratio_treshold


def cross_check():
    """
    """
    return


def feature_matcher(features_query: list, descriptors_query: list, features_train: list, descriptors_train: list) -> list:
    """
    Return list of all matches between two images in form (keypoint_1, keypoint_2)
    which denotes a keypoint from query image and its representative from
    train image respectively
    """
    result = []
    for i in range(len(descriptors_query)):
        nearest_neighbors = knn(descriptors_query[i], descriptors_train)
        if ratio_test(nearest_neighbors):
            result.append(
                (features_query[i], features_train[nearest_neighbors[0][1]]))
    return result
