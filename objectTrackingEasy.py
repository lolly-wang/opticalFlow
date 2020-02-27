'''
  File name: objectTracking.py
  Author:
  Date created:
'''

'''
  File clarification:
    - Input rawVideo: The input video containing one or more objects.
    - Output trackedVideo: The generated output video showing all the tracked features (please do try to show
                           the trajectories for all the features) on the object as well as the bounding boxes.
'''

from scipy import signal
import imageio
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from applyGeometricTransformation import applyGeometricTransformation
from estimateAllTranslation import estimateAllTranslation
from getFeatures import getFeatures
from skimage.feature import corner_shi_tomasi, corner_peaks


def objectTracking(rawVideo):
    frameCount = 0
    if rawVideo.isOpened():
        rval, frame = rawVideo.read()
    else:
        rval = False
        print(rval)

    val = 2
    frames = []
    framesGray=[]
    while rval:
        rval, frame = rawVideo.read()

        # if (frameCount % val == 0):
        # cv2.imwrite('image/' + str(frameCount) + '.jpg', frame)
        try:
            frames.append(frame)
            framesGray.append(utils.rgb2gray(frame))
            # frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        except TypeError:
            break
        # plt.imshow(frame)
        # plt.axis('off')
        # plt.show()
        frameCount = frameCount + 1
    rawVideo.release()
    frameCount = frameCount // val + 1
    print('frameCount : ', frameCount)

    # x1, y1, h1, w1 = 180, 285, 120, 90
    # x2, y2, h2, w2 = 75, 260, 50, 50
    x1, y1, h1, w1 = 290, 185, 80, 110
    x2, y2, h2, w2 = 260, 75, 50, 50
    # bbox = [np.array([[y1, x1], [y1 + h1, x1], [y1, x1 + w1], [y1 + h1, x1 + w1]]),
    #         np.array([[y2, x2], [y2 + h2, x2], [y2, x2 + w2], [y2 + h2, x2 + w2]])]
    bbox = [np.array([[x1, y1], [x1 + w1, y1], [x1, y1 + h1], [x1 + w1, y1 + h1]]),
            np.array([[x2, y2], [x2 + w2, y2], [x2, y2 + h2], [x2 + w2, y2 + h2]])]
    startXs, startYs = getFeatures(framesGray[0], bbox)
    outputFrames = [list() for i in range(150)]

    mask = np.zeros_like(frames[0])

    for i in range(150):  # 100 frames
        print('Processing frame',i+1)
        Xarray, Yarray, XarrayNew, YarrayNew = [], [], [], []
        img1Gray = framesGray[i].copy()
        imgDraw = frames[i].copy()
        img2Gray = framesGray[i].copy()
        for j in range(len(bbox)):
            if bbox[j][3, 0] <= imgDraw.shape[1] or bbox[j][3, 1] <= imgDraw.shape[0]:
                imgDraw = cv2.rectangle(imgDraw, (bbox[j][0,0], bbox[j][0,1]), (bbox[j][3,0], bbox[j][3,1]),
                                    (255, 255, 255),
                                    1)
        # for m in range(len(startXs)):
        #     for n in range(len(startXs[m])):
        #         imgDraw = cv2.circle(imgDraw, (startXs[m][n], startYs[m][n]), 3, (255, 255, 255), thickness=-1)


        startXs, startYs, newXs, newYs, minFeatureNum = estimateAllTranslation(startXs, startYs, img1Gray, img2Gray)

        if  minFeatureNum < 6:
            startXs, startYs = getFeatures(img1Gray, bbox)
            startXs, startYs, newXs, newYs, minFeatureNum = estimateAllTranslation(startXs, startYs, img1Gray, img2Gray)
            print('recalculate features!!!!!!!!!!!!!!!!!!!!!!')

        for f in range(len(bbox)):
            newXs[f], newYs[f] = np.round(newXs[f]).astype(int), np.round(newYs[f]).astype(int)
            startXs[f], startYs[f] = np.round(startXs[f]).astype(int), np.round(startYs[f]).astype(int)

        for k in range(len(startXs)):
            Xarray.extend(startXs[k])
            Yarray.extend(startYs[k])
        for k in range(len(newXs)):
            XarrayNew.extend(newXs[k])
            YarrayNew.extend(newYs[k])

        for ind in Xarray:
            if ind < 0: ind = 0
            if ind > imgDraw.shape[1] - 1: ind = imgDraw.shape[1] - 1
        for ind in Yarray:
            if ind < 0: ind = 0
            if ind > imgDraw.shape[0] - 1: ind = imgDraw.shape[0] - 1
        for ind in XarrayNew:
            if ind < 0: ind = 0
            if ind > imgDraw.shape[1] - 1: ind = imgDraw.shape[1] - 1
        for ind in YarrayNew:
            if ind < 0: ind = 0
            if ind > imgDraw.shape[0] - 1: ind = imgDraw.shape[0] - 1

        for k in range(len(Xarray)):
            mask = cv2.line(mask, (Xarray[k], Yarray[k]), (XarrayNew[k], YarrayNew[k]), (255, 0, 0),2)
        imgDraw = imgDraw - imgDraw * np.array([(mask[:, :, 0] != 0) for m in range(3)]).transpose(1, 2, 0) + mask
        # plt.imshow(imgDraw)
        # plt.show()

        outputFrames[i] = imgDraw


        bbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)

        for f in range(len(bbox)):
            # if bbox[f][3,0]>imgDraw.shape[1] or bbox[f][3,1]>imgDraw.shape[0]:
            #     bbox = [bbox.pop(f)]
            #     newXs, newYs = [newXs.pop(f)], [newYs.pop(f)]
            startXs, startYs = newXs, newYs


    outputFrames=np.array(outputFrames)
    imageio.mimwrite('Optical Flow Easy.mp4',outputFrames, fps=15)
    return


if __name__ == "__main__":
    rawVideo = cv2.VideoCapture('Easy.mp4')
    objectTracking(rawVideo)
