'''
  File name: estimateFeatureTranslation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate Single Feature Translation:
    - Input startX: Represents the starting X coordinate for a single feature in the ﬁrst frame.
    - Input startY: Represents the starting Y coordinate for a single feature in the ﬁrst frame.
    - Input Ix: H ×W matrix representing the gradient along the X-direction
    - Input Iy: H ×W matrix representing the gradient along the Y-direction
    - Input I1: H ×W matrix representing the gray-scale ﬁrst image frame
    - Input I2: H ×W matrix representing the gray-scale second image frame
    - Output newX: Represents the new X coordinate for a single feature in the second frame.
    - Output newY: Represents the new Y coordinate for a single feature in the second frame.
'''

from skimage.feature import corner_harris, corner_peaks
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy


def estimateFeatureTranslation(startX, startY, Ix, Iy, I1, I2):
    w = 20
    Ixw = Ix[startY - w:startY + w + 1, startX - w:startX + w + 1]
    Iyw = Iy[startY - w:startY + w + 1, startX - w:startX + w + 1]
    I1w = I1[startY - w:startY + w + 1, startX - w:startX + w + 1]
    sumIxIx = np.sum(Ixw * Ixw, dtype=np.float64)
    sumIyIy = np.sum(Iyw * Iyw, dtype=np.float64)
    sumIxIy = np.sum(Ixw * Iyw, dtype=np.float64)
    ATA = np.array([[sumIxIx, sumIxIy],
                    [sumIxIy, sumIyIy]])
    try :

        invATA = np.linalg.inv(ATA)
    except np.linalg.LinAlgError:
        invATA = ATA
        print('gggggggggggggggggggggggggggggg')

    w = float(w)
    tmpX = np.arange(startX - w, startX + w + 1.0)
    tmpY = np.arange(startY - w, startY + w + 1.0)

    iter = 10
    tmp = np.zeros((iter, 4))
    interpFunc = interpolate.interp2d(np.array(range(I2.shape[1])), np.array(range(I2.shape[0])), I2, kind='linear')
    u, v, deltU, deltV = 0, 0, 0, 0
    for i in range(iter):
        I2w = interpFunc(u + startX, v + startY)
        It = I2w - I1w
        u += deltU
        v += deltV
        ATAright=np.array([np.sum(Ixw * It), np.sum(Iyw * It)])
        [deltU, deltV] = - np.dot(invATA, np.array([np.sum(Ixw * It), np.sum(Iyw * It)]))
        uv=scipy.sparse.linalg.spsolve(ATA, -ATAright)
        tmp[i, :] = np.array([np.sum(abs(It)), (u ** 2 + v ** 2), u, v])
        # tmp[i, :] = np.array([np.sum(abs(It)), (u ** 2 + v ** 2), u, v])

    # (startX + u), (startY + v)
    displaceTh = 6 ** 2
    indx = np.argmin(tmp[:, 0])
    if tmp[indx, 1] > displaceTh or (startX + tmp[indx, 2]) >= I1.shape[1] or (startY + tmp[indx, 3]) >= I1.shape[0]:
        filStartX = np.nan
        filStartY = np.nan
        newX = np.nan
        newY = np.nan
    else:
        filStartX = startX
        filStartY = startY
        newX = startX + tmp[indx, 2]
        newY = startY + tmp[indx, 3]

    return filStartX, filStartY, newX, newY
