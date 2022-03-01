#!/usr/bin/env python
import cv2
import numpy as np
from undistort import undistort
from gps_simulator import joint_estimation_2
import imutils
# ----------------------------------------------------------------------------------------------------------

class Image_processes:

    def __init__(self):
        self.calibratred = False
        self.undistorter = undistort()
        self.position_estimator = joint_estimation_2()
        return

    def runProcessor(self, frame):
        return self.__imageSegmentation(frame)

    # Perform image processing
    def __imageSegmentation(self, image):

        img = image

        # print(img.shape)  # Print image shape
        cv2.imshow("original", img)

        # get original feed dimentions
        width, height = img.shape[:2]
     

        # Cropping images
        cropped_image1 = self.undistorter.undistort(img[0:int(width / 2), 0:int(height / 2)])
        cropped_image2 = self.undistorter.undistort(img[int(width / 2):width, 0:int(height / 2)])
        cropped_image3 = self.undistorter.undistort(img[0:int(width / 2), int(height / 2):height])
        cropped_image4 = self.undistorter.undistort(img[int(width / 2):width, int(height / 2):height])

        images = [cropped_image1, cropped_image2, cropped_image3, cropped_image4]
        # stitched_img = self.stitcher.main(images)
        # cv2.imshow('stitched images', stitched_img)
        for img in images:
            pos = self.position_estimator.detect_color(img, 'red')
            print(pos)
        cv2.imshow('image1', cropped_image1)
        cv2.imshow('image2', cropped_image2)
        cv2.imshow('image3', cropped_image3)
        cv2.imshow('image4', cropped_image4)
        c = cv2.waitKey(1)

        return

    def __imageStitch(self, images):

        stitcher = cv2.Stitcher_create() 
        (status, stitched) = stitcher.stitch(images)

        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

        # convert the stitched image to grayscale and threshold it
        # such that all pixels greater than zero are set to 255
        # (foreground) while all others remain 0 (background)

        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # the *largest* contour which will be the contour/outline of
        # the stitched image
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # allocate memory for the mask which will contain the
        # rectangular bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        minRect = mask.copy()
        sub = mask.copy()

        # keep looping until there are no non-zero pixels left in the
        # subtracted image
        while cv2.countNonZero(sub) > 0:
            # erode the minimum rectangular mask and then subtract
            # the thresholded image from the minimum rectangular mask
            # so we can count if there are any non-zero pixels left
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

        # find contours in the minimum rectangular mask and then
        # extract the bounding box (x, y)-coordinates

        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        # use the bounding box coordinates to extract the our final
        # stitched image
        stitched = stitched[y:y + h, x:x + w]

        return self.__colourSpaceCoordinate(stitched)

    def __colourSpaceCoordinate(self, image):

        red_u = (20, 20, 256)
        red_l = (0, 0, 100)
        climits = [[red_l, red_u]]

        masks = [cv2.inRange(image, climit[0], climit[1]) for climit in climits]
        maskJs = [cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) for mask in masks]

        frames = [(image & maskJ) for maskJ in maskJs]

        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]

        jThreshes = [cv2.threshold(gray_frame, 1, 255, cv2.THRESH_BINARY) for gray_frame in gray_frames]

        jcontours = [cv2.findContours(jthresh[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) for jthresh in jThreshes]

        cords = []
        radiuslist = []
        for jcontour in jcontours:
            # print(jcontour)
            try:
                Gradius = 0
                (Gx, Gy), Gradius = cv2.minEnclosingCircle(self.mergeContors(jcontour[0]))
                radiuslist.append(Gradius)
                # print(Gradius)
                if Gradius < 2:  # Filter out single pixel showing
                    cords.append([-1, -1])
                else:
                    cords.append([Gx, Gy])

            except:
                cords.append([-1, -1])
                radiuslist.append(0)

        contourDic = {"Red": {'x': cords[3][0], 'y': cords[3][1]}}

        im_copy = image.copy()

        for i in range(len(cords)):
            cv2.circle(im_copy, (int(cords[i][0]), int(cords[i][1])), 2, (255, 255, 255), -1)
            cv2.putText(im_copy, list(contourDic.keys())[i], (int(cords[i][0]) - 50, int(cords[i][1]) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(im_copy, (int(cords[i][0]), int(cords[i][1])), int(radiuslist[i]), (0, 255, 0), 1)

        return contourDic, im_copy

    def __mergeContors(self, ctrs):
        list_of_pts = []
        for c in ctrs:
            for e in c:
                list_of_pts.append(e)
        ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
        ctr = cv2.convexHull(ctr)
        return ctr

    def sdpPixelToDegrees(self):
        return
