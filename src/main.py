import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
cv2.ocl.setUseOpenCL(False)


"""
TBD
Description
"""

FEATURE_EXTRACTOR = "SIFT"
FEATURE_TO_MATCH = "KNN"

"""
Importing train (image that needs to be transformed) and query images
"""

train_image = cv2.imread(
    "dataset/hotel/hotel-05.png")   # Returns in GBR
train_image = cv2.cvtColor(
    train_image, cv2.COLOR_BGR2RGB)   # Converting to RGB
# train_image = imutils.rotate_bound(train_image, 180) # Just 4 fun
train_image_bw = cv2.cvtColor(
    train_image, cv2.COLOR_RGB2GRAY)  # Making black & white


query_image = cv2.imread(
    "dataset/hotel/hotel-06.png")   # Returns in GBR
query_image = cv2.cvtColor(
    query_image, cv2.COLOR_BGR2RGB)   # Converting to RGB
query_image_bw = cv2.cvtColor(
    query_image, cv2.COLOR_RGB2GRAY)  # Making black & white


# Plotting provided images

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=False, figsize=[16, 9])

ax1.imshow(query_image, cmap="gray")
ax1.set_xlabel("Query image", fontsize=14)

ax2.imshow(train_image, cmap="gray")
ax2.set_xlabel("Train image", fontsize=14)

plt.show()


"""
PART 1 – Keypoints and features extraction
"""


def select_descriptor_methods(image, method):

    assert method is not None, "Please define a feature descriptor method. accepted Values are: 'sift', 'surf'"

    if method == 'SIFT':
        descriptor = cv2.SIFT_create()
    elif method == 'SURF':
        descriptor = cv2.SURF_create()
    elif method == 'BRISK':
        descriptor = cv2.BRISK_create()
    elif method == 'ORB':
        descriptor = cv2.ORB_create()

    (keypoints, features) = descriptor.detectAndCompute(image, None)

    return (keypoints, features)


keypoints_train_img, features_train_img = select_descriptor_methods(
    train_image_bw, FEATURE_EXTRACTOR)

keypoints_query_img, features_query_img = select_descriptor_methods(
    query_image_bw, FEATURE_EXTRACTOR)

# Checking what we've got
# print("Keypoints: ", keypoints_train_img)
# print("Features: ", features_train_img)

# Checking what keypoint consists of
for keypoint in keypoints_query_img:
    x, y = keypoint.pt
    size = keypoint.size
    orientation = keypoint.angle
    response = keypoint.response
    octave = keypoint.octave
    class_id = keypoint.class_id
# print("x, y ", x, y)
# print("size ", size)
# print("orientation ", orientation)
# print("response ", response)
# print("octave ", octave)
# print("class_id ", class_id)

# Checkong that descriptors is a numpy array of shape (Number of Keypoints)×128
# print(len(keypoints_query_img))
# print(features_query_img.shape)


# Displaying the keypoints and features detected on both images

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                               figsize=(20, 8), constrained_layout=False)

ax1.imshow(cv2.drawKeypoints(train_image_bw,
           keypoints_train_img, None, color=(0, 255, 0)))

ax1.set_xlabel("(a)", fontsize=14)

ax2.imshow(cv2.drawKeypoints(query_image_bw,
           keypoints_query_img, None, color=(0, 255, 0)))
ax2.set_xlabel("(b)", fontsize=14)

plt.savefig("./output/" + FEATURE_EXTRACTOR + "_features_img_"+'.jpeg', bbox_inches='tight',
            dpi=300, format='jpeg')
plt.show()


"""
PART 2 – Matching features (Brute force matcher)
"""


def create_matching_object(method, crossCheck):
    "Create and return a Matcher Object"

    # For BF matcher, first we have to create the BFMatcher object using cv2.BFMatcher().
    # It takes two optional params.
    # normType - It specifies the distance measurement
    # crossCheck - which is false by default. If it is true, Matcher returns only those matches
    # with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match
    # and vice-versa.
    if method == 'SIFT' or method == 'SURF':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'ORB' or method == 'BRISK':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def key_points_matching(features_train_img, features_query_img, method):
    bf = create_matching_object(method, crossCheck=True)

    # Match descriptors.
    best_matches = bf.match(features_train_img, features_query_img)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    print("Raw matches with Brute force: ", len(rawMatches))
    return rawMatches


def key_points_matching_KNN(features_train_img, features_query_img, ratio, method):
    bf = create_matching_object(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(features_train_img, features_query_img, k=2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m, n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


# Printing matched points between images

print(f"Drawing: {FEATURE_TO_MATCH} matched features Lines")

fig = plt.figure(figsize=(20, 8))

if FEATURE_TO_MATCH == 'BF':
    matches = key_points_matching(
        features_train_img, features_query_img, method=FEATURE_EXTRACTOR)

    mapped_features_image = cv2.drawMatches(train_image, keypoints_train_img, query_image, keypoints_query_img, matches[:100],
                                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Now for cross checking draw the feature-mapping lines also with KNN
elif FEATURE_TO_MATCH == 'KNN':
    matches = key_points_matching_KNN(
        features_train_img, features_query_img, ratio=0.75, method=FEATURE_EXTRACTOR)

    mapped_features_image = cv2.drawMatches(train_image, keypoints_train_img, query_image, keypoints_query_img, np.random.choice(matches, 100),
                                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


plt.imshow(mapped_features_image)
plt.axis('off')
plt.savefig("./output/" + FEATURE_EXTRACTOR + "_matching_img_"+'.jpeg', bbox_inches='tight',
            dpi=300, optimize=True, format='jpeg')
plt.show()


"""
PART 3 – Constructing homography (RANSAC)
"""


def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):
    """
    Converting the keypoints to numpy arrays before passing them for calculating Homography Matrix.
    Because we are supposed to pass 2 arrays of coordinates to cv2.findHomography, as in I have these points in image-1, and I have points in image-2, so now what is the homography matrix to transform the points from image 1 to image 2
    """
    keypoints_train_img = np.float32(
        [keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32(
        [keypoint.pt for keypoint in keypoints_query_img])

    '''
    For findHomography() - I need to have an assumption of a minimum of correspondence points that are present between the 2 images. Here, I am assuming that Minimum Match Count to be 4
    '''
    if len(matches) > 4:
        # construct the two sets of points
        points_train = np.float32(
            [keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32(
            [keypoints_query_img[m.trainIdx] for m in matches])

        # Calculate the homography between the sets of points
        (H, status) = cv2.findHomography(
            points_train, points_query, cv2.RANSAC, reprojThresh)

        return (matches, H, status)
    else:
        return None


M = homography_stitching(
    keypoints_train_img, keypoints_query_img, matches, reprojThresh=4)

if M is None:
    print("Error!")

(matches, Homography_Matrix, status) = M

print(Homography_Matrix)


# For the calculation of the width and height of the final horizontal panoramic images
# I can just add the widths of the individual images and for the height
# I can take the max from the 2 individual images.

width = query_image.shape[1] + train_image.shape[1]
print("width ", width)


height = max(query_image.shape[0], train_image.shape[0])

# otherwise, apply a perspective warp to stitch the images together

# Now just plug that "Homography_Matrix"  into cv::warpedPerspective and I shall have a warped image1 into image2 frame

result = cv2.warpPerspective(train_image, Homography_Matrix,  (width, height))

# The warpPerspective() function returns an image or video whose size is the same as the size of the original image or video. Hence set the pixels as per my query_photo

result[0:query_image.shape[0], 0:query_image.shape[1]] = query_image

plt.figure(figsize=(20, 10))
plt.axis('off')
plt.imshow(result)

imageio.imwrite("./output/horizontal_panorama_img_"+'.jpeg', result)

plt.show()
