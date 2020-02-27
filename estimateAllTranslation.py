'''
  File name: estimateAlltranslation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate Single Feature Translation:
    - Input startXs: Represents the starting X coordinate of all the features in the ﬁrst frame for all the bounding boxes.
    - Input startYs: Represents the starting Y coordinate of all the features in the ﬁrst frame for all the bounding boxes.
    - Input I1: H ×W  matrix representing the gray-scale ﬁrst image frame
    - Input I2: H ×W  matrix representing the gray-scale second image frame
    - Output newXs: Represents the new X coordinate of all the features in all the bounding boxes.
    - Output newYs: Represents the new Y coordinate of all the features in all the bounding boxes.
'''

import numpy as np
from scipy import interpolate
import utils
from scipy import signal
from estimateFeatureTranslation import estimateFeatureTranslation
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy

# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def findDerivatives(I_gray):
    G = utils.GaussianPDF_2D(0, 1, 9, 9)  # window > 2*sigma
    Gx, Gy = np.gradient(G, axis=(1, 0))
    magx = signal.convolve2d(I_gray, Gx, 'same')
    magy = signal.convolve2d(I_gray, Gy, 'same')
    # sobelx = cv2.Sobel(I_gray/255, cv2.CV_64F, 1, 0, ksize=9)
    # sobely = cv2.Sobel(I_gray/255, cv2.CV_64F, 0, 1, ksize=9)
    # sx = ndimage.sobel(I_gray, axis=0, mode='constant')
    # sy = ndimage.sobel(I_gray, axis=1, mode='constant')
    return magx, magy


def estimateAllTranslation(startXs, startYs, I1, I2):
    Ix, Iy = findDerivatives(I1)
    featureNum = []
    newXs, newYs, filStartXs, filStartYs = [], [], [], []
    for i in range(len(startXs)):  # search all the bounding boxes
        tmpX = np.zeros_like(startXs[i], dtype=float)
        tmpY = np.zeros_like(startXs[i], dtype=float)
        filStartX = np.zeros_like(startXs[i], dtype=float)
        filStartY = np.zeros_like(startXs[i], dtype=float)

        for j in range(len(startXs[i])):
            filStartX[j], filStartY[j], tmpX[j], tmpY[j] = estimateFeatureTranslation(startXs[i][j], startYs[i][j], Ix,
                                                                                      Iy, I1, I2)
        filStartX = filStartX[~np.isnan(tmpX)]
        filStartY = filStartY[~np.isnan(tmpX)]
        tmpX = tmpX[~np.isnan(tmpX)]
        tmpY = tmpY[~np.isnan(tmpY)]
        filStartXs.append(filStartX)
        filStartYs.append(filStartY)
        newXs.append(tmpX)
        newYs.append(tmpY)
        featureNum.append(len(tmpX))
    minFeatureNum = min(featureNum)
    return filStartXs, filStartYs, newXs, newYs, minFeatureNum
