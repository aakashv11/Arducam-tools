import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2
import fisheye_correction

record_time = 15

camera_resolution = [640,480]
camera_framerate = 20

fourcc = cv2.VideoWriter_fourcc(*'MP42')
video = cv2.VideoWriter('demo.avi', fourcc, camera_framerate, tuple(camera_resolution))

camera = PiCamera()
camera.resolution = tuple(camera_resolution)
camera.framerate = camera_framerate
rawCapture = PiRGBArray(camera, size=tuple(camera_resolution))

start_time = time.time()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    # Comment this out if your camera is mounted on the top
    image = cv2.flip(image, -1)

    # Correct fisheye distortion (should NOT be corrected for using with ORBSLAM3)
    image = fisheye_correction.undistort(image)

    video.write(image)

    rawCapture.truncate(0)

    if time.time() - start_time > record_time:
        break

video.release()

