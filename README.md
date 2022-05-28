# Image Stitching – Linear Algebra @ UCU course project

In the digital age image stitching proves itself useful in many areas of our lives, from science to leisure. In this project the authors discuss the approaches to image stitching by using linear algebra algorithms such as SIFT, KNN and RANSAC. The aim of the project is to develop a tool for creating high-quality images with a larger aspect ratio as a replacement for the widely popular real-time stitching solutions.
The authors implemented an application for this purpose on their own with the help of the OpenCV library in Python. The execution yielded successful results when tested on inputs of varying size and quality.

### Components

1. `main.py` – main module of the app which performs stitching
2. `src/stitcher.py` – the module that stitches images using high-level API from OpenCV
3. `src/features_matcher.py` – the module that has the implementation of the functionality for performing KNN algorithm with Lowe's ratio test for matching keypoints
4. `src/ransac.py` – the module that has the implementation of the functionality for performing homography estimation with RANSAC
5. `src/utils.py` – the module that has the implementation of auxillary functions, such as parsing image files, blending and cropping images etc.

### Usage

Create virtual python environment and install requirements from `requirements.txt`.

Prepare a directory with images to stitch. Preferrably to rename images so that the central is 1, and each on the left or right +1. Like: 2 - 1 - 3 (that would increase stitching quality).

Example of program run:

```shell
[user@pc image-stitching]$ python3 main.py ./dataset/UCU

---- IMAGE STITCHER ----

3 images detected

Stitching images 1 and 2


Start matching features
End matching features

Calculating homography for image №1...

Stitching image №1...

Stitching images 2 and 3


Start matching features
End matching features

Calculating homography for image №2...

Stitching image №2...


DONE!
```

After a successfull image stitching, the stitched photo will apear on the screen. Press any key to close the window with the photo and exit the program.

### Authors (team):

- [Roman Kulyk](https://github.com/kkulykk)

- [Bohdan Mykhailiv](https://github.com/bmykhaylivvv)

- [Olesia Nedopas](https://github.com/Lesi-N)
