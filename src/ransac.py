import numpy as np
from math import log, sqrt

import random


def calculate_homography(homography_base):
    k = len(homography_base)
    cmatrix = [[0 for _ in range(k*2)] for _ in range(k*2)]
    query_vec = [0] * (k*2)
    print('matches: ')
    for i in range(k):
        itv = 2 * i
        train_x, train_y = homography_base[i][0].pt
        query_x, query_y = homography_base[i][1].pt
        print(homography_base[i][0].pt, homography_base[i][1].pt)

        cmatrix[itv][:3] = train_x, train_y, 1
        cmatrix[itv][6:] = train_x * query_x * (-1), train_y * query_x * (-1)
        cmatrix[itv + 1][3:6] = train_x, train_y, 1
        cmatrix[itv + 1][6:] = train_x * query_y * (-1), train_y * query_y * (-1)
        query_vec[itv], query_vec[itv + 1] = query_x, query_y

    hom_coef = np.linalg.solve(cmatrix, query_vec)
    hom_coef = np.reshape(np.append(hom_coef, [1]), (3, 3))
    return hom_coef


def count_inliers(homography, datapoints, num_datapoints, threshold):
    inliers = 0
    for i in range(num_datapoints):
        match = datapoints[i]
        init_pos = np.array([match[0].pt[0], match[0].pt[1], 1])

        expected_pos = homography.dot(init_pos)
        vec = [expected_pos[0] - match[1].pt[0], expected_pos[1] - match[1].pt[1]]
        dist = sqrt(vec[0]**2 + vec[1]**2)

        if dist < threshold:
            inliers += 1

    return inliers


def ransac(matched_points, sample_size, eps, inlier_prob, desired_prob):
    max_inlier_fit = [[], 0]
    num_matches = len(matched_points)
    acceptable_fit = desired_prob * num_matches
    N_iterations = int(log(1 - desired_prob) / log(1 - inlier_prob ** sample_size))

    for i in range(N_iterations):
        sample_idx = random.sample(range(1, num_matches), 4)
        rand_samp = [matched_points[i] for i in sample_idx]

        sample_fit = calculate_homography(rand_samp)

        inliers = count_inliers(sample_fit, matched_points, num_matches, eps)

        if inliers >= acceptable_fit:
            return sample_fit

        if inliers > max_inlier_fit[1]:
            max_inlier_fit = [sample_fit, inliers]

    return max_inlier_fit
