#!/usr/bin/env python

"""
From https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import sys
import os
import json


#---------------------- SET THE PARAMETERS
nRows = 9
nCols = 6
dimension = 25 #- mm

for i in range(len(sys.argv)):
    if sys.argv[i] == '-r':
        nRows = int(sys.argv[i + 1])
    if sys.argv[i] == '-c':
        nCols = int(sys.argv[i+1])
    if sys.argv[i] == '-d':
        dimension = float(sys.argv[i + 1])

workingFolder   = "./images"
imageType       = 'jpg'
#------------------------------------------

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1, nRows*nCols, 3), np.float32)
objp[0,:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Find the images files
filename    = workingFolder + "/*." + imageType
images      = glob.glob(filename)

print("Found", len(images), "images")
if len(images) < 15:
    print("Not enough images were provided. We need at least 15 images for a" \
          " good calibration attempt")
    sys.exit()

else:
    nPatternFound = 0
    img_to_undistort = images[0]

    for fname in images:
        if 'calibresult' in fname: continue
        #-- Read the file and convert in greyscale
        img     = plt.imread(os.path.join(os.getcwd(), fname[2:]))
        gray    = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        print("Reading image ", fname)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Pattern found! Press ESC to skip or ENTER to accept")
            #--- Sometimes, Harris corners fails with crappy pictures, so
            corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nCols,nRows), corners2,ret)
            cv2.imshow('img',img)
            # cv2.waitKey(0)
            k = cv2.waitKey(0) & 0xFF
            if k == 27: #-- ESC Button
                print("Image Skipped")
                img_to_undistort = fname
                continue

            print("Image accepted")
            nPatternFound += 1
            objpoints.append(objp)
            imgpoints.append(corners2)

            # cv2.waitKey(0)
        else:
            img_to_undistort = fname


cv2.destroyAllWindows()

if (nPatternFound > 9):
    print("Found %d good images" % (nPatternFound))
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    # print("Found " + str(N_OK) + " valid images for calibration")
    
    cv2.imwrite(workingFolder + "/calibresult.jpg",dst)
    print("Calibrated picture saved as calibresult.jpg")
    print("Calibration Matrix:")
    print(K)
    print("Disortion:", D)

    #--------- Save result in json file
    # make camera matrix and distortion vector into dictionary
    K2 = K.tolist()
    D2 = D.tolist()
    data = {
        "intrinsics": K2,
        "distortion": D2
    }
    with open('cameraInfo.json', 'w') as json_file:
        json.dump(data, json_file)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error/len(objpoints))

    # Undistort an image
    img = plt.imread(os.path.join(os.getcwd(), img_to_undistort[2:]))
    img = img[..., ::-1]  # RGB --> BGR
    # h,  w = img.shape[:2]
    print("Image to undistort: ", img_to_undistort)

    h, w = img.shape[:2]
    newcameramtx=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,(w,h),np.eye(3),balance=0.5)

    mapx,mapy = cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),newcameramtx,(w,h), cv2.CV_16SC2) #newcameramtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    print("DIM=" + str(img.shape))
    print("K=np.array(" + str(K) + ")")
    print("D=np.array(" + str(D) + ")")

else:
    print("No calibration occurred because less than 10 images had correct patterns")
    print("In order to calibrate you need at least 10 correct pattern identifications")
    print("Take some more pictures and try again")

