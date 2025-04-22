import cv2
import numpy as np
import utils

webCam = False
path = 'test_img_3.jpg'

capture = cv2.VideoCapture(0) # id = 0
capture.set(10, 160)
capture.set(3, 3072)
capture.set(4, 4096)
scale = 2
wPap = 210 * scale
hPap = 297 * scale

while True:
    if webCam:
        success,img = capture.read()
    else:
        img = cv2.imread(path)
        img = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    img, finalContours = utils.getContours(img, minArea=50000, filter=4)

    if len(finalContours) != 0:
        biggest = finalContours[0][2]
        imgWarp = utils.warpImg(img, biggest, wPap, hPap)
        imgContours2, cont2 = utils.getContours(imgWarp, minArea=2000, filter=4, cThr = [50,50])

        if len(cont2) != 0:
            for obj in cont2:
                cv2.polylines(imgContours2, [obj[2]], True, (0,255,0), 2)
                nPoints = utils.reorder(obj[2])
                nWid = round((utils.findDistance(nPoints[0][0]//scale, nPoints[1][0]//scale))/10,1)
                nHei = round((utils.findDistance(nPoints[0][0]//scale, nPoints[2][0]//scale))/10,1)

                # arrow with measurements
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),(255, 0, 255), 2, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),(255, 0, 255), 2, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{} cm'.format(nWid), (x+30, y+5), cv2.FONT_HERSHEY_SIMPLEX, .5, (23, 255, 23), 2)
                cv2.putText(imgContours2, '{} cm'.format(nHei), (x-30, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (23, 255, 23), 2)


        cv2.imshow('A4', imgContours2)

    cv2.imshow('Original', img)
    cv2.waitKey(1)