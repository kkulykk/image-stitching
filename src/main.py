import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import ransac
import feature_matcher

cv2.ocl.setUseOpenCL(False)


"""
TBD
Description
"""

"""
Importing train (image that needs to be transformed) and query images
"""

DEBUG = False

train_image = cv2.imread(
    "dataset/hotel/hotel-02.png")
train_image = cv2.cvtColor(
    train_image, cv2.COLOR_BGR2RGB)
train_image_bw = cv2.cvtColor(
    train_image, cv2.COLOR_RGB2GRAY)


query_image = cv2.imread(
    "dataset/hotel/hotel-03.png")   # Returns in GBR
query_image = cv2.cvtColor(
    query_image, cv2.COLOR_BGR2RGB)   # Converting to RGB
query_image_bw = cv2.cvtColor(
    query_image, cv2.COLOR_RGB2GRAY)  # Making black & white


if DEBUG:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, constrained_layout=False, figsize=[16, 9])

    ax1.imshow(query_image, cmap="gray")
    ax1.set_xlabel("Query image", fontsize=14)

    ax2.imshow(train_image, cmap="gray")
    ax2.set_xlabel("Train image", fontsize=14)


"""
PART 1 – Keypoints and features extraction
"""


sift = cv2.SIFT_create()

keypoints_train_img, features_train_img = sift.detectAndCompute(
    train_image, None)

keypoints_query_img, features_query_img = sift.detectAndCompute(
    query_image, None)


if DEBUG:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(20, 8), constrained_layout=False)

    ax1.imshow(cv2.drawKeypoints(train_image,
                                 keypoints_train_img, None, color=(0, 255, 0)))

    ax1.set_xlabel("(a)", fontsize=14)

    ax2.imshow(cv2.drawKeypoints(query_image,
                                 keypoints_query_img, None, color=(0, 255, 0)))
    ax2.set_xlabel("(b)", fontsize=14)


"""
PART 2 – Matching features (Brute force matcher)
"""

mtchs = feature_matcher.feature_matcher(
    list(keypoints_query_img), list(features_query_img), list(keypoints_train_img), list(features_train_img))


"""
PART 3 – Constructing homography (RANSAC)
"""


print("Calculating homography...")
H = ransac.ransac(mtchs, 4, 5, 0.7, 0.95)
print(H)


width = query_image.shape[1] + train_image.shape[1]


height = max(query_image.shape[0], train_image.shape[0])

print("Warping image...")

result = cv2.warpPerspective(train_image, H,  (width, height))

result[0:query_image.shape[0], 0:query_image.shape[1]] = query_image

plt.figure(figsize=(20, 10))
plt.axis('off')
plt.imshow(result)

imageio.imwrite(
    "img"+'.jpeg', result)

plt.show()