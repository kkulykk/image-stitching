import os
import numpy as np
import cv2


"""
Some of the functions for appropriate overlap blending (slightly modified) were taken
from https://www.kaggle.com/code/deepzsenu/multiple-image-stitching/notebook
"""


def convertResult(img):
    '''Because of our images which were loaded by opencv, 
    in order to display the correct output with matplotlib, 
    you need to reduce the range of your floating point image from [0,255] to [0,1] 
    and converting the image from BGR to RGB:'''
    img = np.array(img, dtype=float)/float(255)
    img = img[:, :, ::-1]
    return img


def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    assert barrier < width
    mask = np.zeros((height, width))

    offset = int(smoothing_window/2)
    try:
        if left_biased:
            mask[:, barrier-offset:barrier+offset +
                 1] = np.tile(np.linspace(1, 0, 2*offset+1).T, (height, 1))
            mask[:, :barrier-offset] = 1
        else:
            mask[:, barrier-offset:barrier+offset +
                 1] = np.tile(np.linspace(0, 1, 2*offset+1).T, (height, 1))
            mask[:, barrier+offset:] = 1
    except:
        if left_biased:
            mask[:, barrier-offset:barrier+offset +
                 1] = np.tile(np.linspace(1, 0, 2*offset).T, (height, 1))
            mask[:, :barrier-offset] = 1
        else:
            mask[:, barrier-offset:barrier+offset +
                 1] = np.tile(np.linspace(0, 1, 2*offset).T, (height, 1))
            mask[:, barrier+offset:] = 1

    return cv2.merge([mask, mask, mask])


def panoramaBlending(dst_img_rz, src_img_warped, width_dst, side, showstep=False):
    '''Given two aligned images @dst_img and @src_img_warped, and the @width_dst is width of dst_img 
    before resize, that indicates where there is the discontinuity between the images, 
    this function produce a smoothed transient in the overlapping.
    @smoothing_window is a parameter that determines the width of the transient
    left_biased is a flag that determines whether it is masked the left image,
    or the right one'''

    h, w, _ = dst_img_rz.shape
    smoothing_window = int(width_dst/8)
    barrier = width_dst - int(smoothing_window/2)
    mask1 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=True)
    mask2 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=False)

    if side == 'left':
        dst_img_rz = cv2.flip(dst_img_rz, 1)
        src_img_warped = cv2.flip(src_img_warped, 1)
        dst_img_rz = (dst_img_rz*mask1)
        src_img_warped = (src_img_warped*mask2)
        pano = src_img_warped+dst_img_rz
        pano = cv2.flip(pano, 1)
    else:
        dst_img_rz = (dst_img_rz*mask1)
        src_img_warped = (src_img_warped*mask2)
        pano = src_img_warped+dst_img_rz

    return pano


def crop(panorama, h_dst, conners):
    '''crop panorama based on destination.
    @param panorama is the panorama
    @param h_dst is the hight of destination image
    @param conner is the tuple which containing 4 conners of warped image and 
    4 conners of destination image'''
    # find max min of x,y coordinate
    [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(conners.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    conners = conners.astype(int)

    # conners[0][0][0] is the X coordinate of top-left point of warped image
    # If it has value<0, warp image is merged to the left side of destination image
    # otherwise is merged to the right side of destination image
    if conners[0][0][0] < 0:
        n = abs(-conners[1][0][0]+conners[0][0][0])
        panorama = panorama[t[1]:h_dst+t[1], n:, :]
    else:
        if(conners[2][0][0] < conners[3][0][0]):
            panorama = panorama[t[1]:h_dst+t[1], 0:conners[2][0][0], :]
        else:
            panorama = panorama[t[1]:h_dst+t[1], 0:conners[3][0][0], :]
    return panorama


def warpTwoImages(src_img, dst_img, H, showstep=False):

    # generate Homography matrix

    # get height and width of two images
    height_src, width_src = src_img.shape[:2]
    height_dst, width_dst = dst_img.shape[:2]

    # extract conners of two images: top-left, bottom-left, bottom-right, top-right
    pts1 = np.float32([[0, 0], [0, height_src], [width_src, height_src], [
                      width_src, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, height_dst], [width_dst, height_dst], [
                      width_dst, 0]]).reshape(-1, 1, 2)

    try:
        # aply homography to conners of src_img
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1_, pts2), axis=0)

        # find max min of x,y coordinate
        [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]

        # top left point of image which apply homography matrix, which has x coordinate < 0, has side=left
        # otherwise side=right
        # source image is merged to the left side or right side of destination image
        if(pts[0][0][0] < 0):
            side = 'left'
            width_pano = width_dst+t[0]
        else:
            width_pano = int(pts1_[3][0][0])
            side = 'right'
        height_pano = ymax-ymin

        # Translation
        # https://stackoverflow.com/a/20355545
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        src_img_warped = cv2.warpPerspective(
            src_img, Ht.dot(H), (width_pano, height_pano))
        # generating size of dst_img_rz which has the same size as src_img_warped
        dst_img_rz = np.zeros((height_pano, width_pano, 3))
        if side == 'left':
            dst_img_rz[t[1]:height_src+t[1], t[0]:width_dst+t[0]] = dst_img
        else:
            dst_img_rz[t[1]:height_src+t[1], :width_dst] = dst_img

        # blending panorama
        pano = panoramaBlending(
            dst_img_rz, src_img_warped, width_dst, side, showstep=showstep)

        # croping black region
        pano = crop(pano, height_dst, pts)
        return pano
    except:
        raise Exception("Please try again with another image set!")


def get_images(path):
    images = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if os.path.join(dirname, filename).endswith(".png") or os.path.join(dirname, filename).endswith(".jpeg"):
                image = cv2.imread(os.path.join(dirname, filename))
                image = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB)
                images.append(image)
    if len(images) < 2:
        raise ValueError("Not enough images to stitch")
    print(f"{len(images)} images detected\n")
    return images
