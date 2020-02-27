'''
  File name: getFeaturesInBox.py
  Author:
  Date created:
'''

'''
  File clarification:
    Identify features within the bounding box for each object:
    - Input img: H ×W matrix representing the grayscale input image.
    - Input bbox: F ×4×2 matrix representing the four corners of the bounding box where F is the number of
                  objects you would like to track.
    - Output x: N ×F matrix representing the N row coordinates of the features across F objects.
    - Output y: N ×F matrix representing the N column coordinates of the features across F objects.
'''

import cv2
import numpy as np


def getFeatures(img, bbox):
    F = len(bbox)
    startXs = [list() for i in range(F)]
    startYs = [list() for i in range(F)]
    for i in range(len(bbox)):
        if bbox[i][0][0]<0:bbox[i][0][0]=0
        if bbox[i][3][0]<0:bbox[i][3][0]=0
        if bbox[i][0][1]<0:bbox[i][0][1]=0
        if bbox[i][3][1]<0:bbox[i][3][1]=0

        imgPatch = img[bbox[i][0][1]:bbox[i][3][1], bbox[i][0][0]:bbox[i][3][0]]
        corners = cv2.goodFeaturesToTrack(imgPatch.astype(np.float32), 50, 1e-3, 0)
        try:
            corners = np.int0(corners)
        except TypeError:
            print('......................................')
        for corner in corners:
            x, y = corner.ravel()
            startXs[i].append(x + bbox[i][0][0])
            startYs[i].append(y + bbox[i][0][1])
        print('get ' + str(len(startYs[i]))+' features in box')

    return startXs, startYs
