import cv2
import os
import imageio
import glob
import numpy as np
from src import ransac
from src import utils
from src import feature_matcher
import argparse

cv2.ocl.setUseOpenCL(False)


"""
Image stitcher module.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image stitcher')
    parser.add_argument('path', metavar='path', type=str, default="./",
                        help='A relative path to the directory with images to stitch')

    args = parser.parse_args()
    print("\n---- IMAGE STITCHER ----\n")
    path = args.path
    images = utils.get_images(path)

    initial_image = images[0]
    if not os.path.exists("./output"):
        os.makedirs("./output")
    else:
        files = glob.glob("./output/*")
        for f in files:
            os.remove(f)

    try:
        for i in range(len(images)-1):
            sift = cv2.SIFT_create()

            keypoints_1, features_1 = sift.detectAndCompute(
                initial_image, None)
            keypoints_2, features_2 = sift.detectAndCompute(
                images[i + 1], None)

            print(f"Stitching images {i+1} and {i+2}\n")
            matches = feature_matcher.feature_matcher(list(keypoints_2), list(
                features_2), list(keypoints_1), list(features_1))

            print(f"Calculating homography for image №{i+1}...\n")

            H = ransac.ransac(matches, 4, 5, 0.7, 0.95)
            print(f"Stitching image №{i+1}...\n")
            result = utils.warpTwoImages(initial_image, images[i+1], H, True)
            imageio.imwrite(os.path.join("./output", f"img{i}.jpeg"),
                            (utils.convertResult(result)*255).astype(np.uint8))
            initial_image = cv2.imread(
                os.path.join("./output", f"img{i}.jpeg"))
        print("\nDONE!\n")
        result_show = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imshow("result.jpeg", utils.convertResult(result_show))
        imageio.imwrite(os.path.join("./output", "result.jpeg"),
                        (utils.convertResult(result)*255).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        raise ValueError("An error occured. Try stitching another images")
