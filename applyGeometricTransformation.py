import math
import numpy as np
import matplotlib.pyplot as plt
from ransac_est_homography import ransac_est_homography

import skimage


def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    newbbox = []
    for i in range(len(bbox)):  # search all the bounding boxes

        # print('old bbox' + str(i))
        # print(bbox[i])

        # n = len(bbox[i])
        ones = np.ones((1, len(bbox[i])))
        transpose = np.array(bbox[i]).T
        tmpBbox = np.concatenate((transpose, ones), axis=0)

        # inlier_ind=ransac_est_homography(startXs[i], startYs[i],newXs[i], newYs[i],0.2)
        #
        # src = np.array([startXs[i][inlier_ind], startYs[i][inlier_ind]])
        # dst = np.array([newXs[i][inlier_ind], newYs[i][inlier_ind]])
        src = np.array([startXs[i], startYs[i]])
        dst = np.array([newXs[i], newYs[i]])

        print('bbox ' + str(i) + ' left ' + str(len(startXs[i])) + ' features')

        tform = skimage.transform.estimate_transform('similarity', src.T, dst.T)
        tformp = np.asmatrix(tform.params)  # transform matrix
        tmpNewbbox = tformp.dot(tmpBbox)
        tmpNewbbox = np.round(tmpNewbbox).astype(int)

        minX = np.amin(tmpNewbbox[0, :])
        maxX = np.amax(tmpNewbbox[0, :])
        minY = np.amin(tmpNewbbox[1, :])
        maxY = np.amax(tmpNewbbox[1, :])


        tmp = np.array([[minX, minY],
                        [maxX, minY],
                        [minX, maxY],
                        [maxX, maxY]])

        # print('new bbox' + str(i))
        # print(tmp)

        newbbox.append(tmp)

    return newbbox
