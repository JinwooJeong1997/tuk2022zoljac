import numpy as np
import math

## Image Processing libraries
import skimage
from skimage import exposure

import scipy.misc as misc
import cv2

import imageio

## Visual and plotting libraries
import matplotlib.pyplot as plt

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def basic_showImg(img):
    '''Shows an image in a numpy.array type. Syntax:
        basic_showImg(img, size=4), where
            img = image numpy.array;
            size = the size to show the image. Its value is 4 by default.
    '''
    plt.figure(figsize=(10,9))
    plt.imshow(img)
    plt.show()

import rawpy
import imageio

raw = rawpy.imread('90.ARW')

im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=16)
rgb = np.float32(im / 65535.0*255.0)
rgb = np.asarray(rgb,np.uint8)

imageio.imsave('image.jpg', rgb)

gray=grayscale(rgb)

imageio.imsave('gray.jpg',gray)
