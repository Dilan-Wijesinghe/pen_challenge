## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# Imports
from __future__ import print_function
import argparse
from copy import deepcopy
from turtle import color
# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2 as cv
import arm

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
cfg = pipeline.start(config) # cfg is a better name for this!
profile = cfg.get_stream(rs.stream.color)
intr = profile.as_video_stream_profile().get_intrinsics() # Intermediates for deproject

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = cfg.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Trackbar Code

# ax_value = 255
max_value_H = 360//2
low_H = 80
# low_S = 0
# low_V = 0
high_H = max_value_H
# high_S = max_value
# high_V = max_value
# window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
# low_S_name = 'Low S'
# low_V_name = 'Low V'
high_H_name = 'High H'
# high_S_name = 'High S'
# high_V_name = 'High V'

# ## [low]
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
# ## [low]

# ## [high]
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
# ## [high]

# def on_low_S_thresh_trackbar(val):
#     global low_S
#     global high_S
#     low_S = val
#     low_S = min(high_S-1, low_S)
#     cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

# def on_high_S_thresh_trackbar(val):
#     global low_S
#     global high_S
#     high_S = val
#     high_S = max(high_S, low_S+1)
#     cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

# def on_low_V_thresh_trackbar(val):
#     global low_V
#     global high_V
#     low_V = val
#     low_V = min(high_V-1, low_V)
#     cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

# def on_high_V_thresh_trackbar(val):
#     global low_V
#     global high_V
#     high_V = val
#     high_V = max(high_V, low_V+1)
#     cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
cv.namedWindow(window_detection_name)
## [window]

## [trackbar]
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
# cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
# cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
# cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
# cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
## [trackbar]

# ------ Robot Arm ------
# Initialize the Robot Arm, so we can move it later
RoboMoves = arm.RobotMovement()
RoboMoves.GoSleep()

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        c_img_for_conts = color_image # Copy of Color Image

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        # depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        # images = np.hstack((bg_removed, depth_colormap))

        # Fixed Thresholding
        LOW_H = 118
        HIGH_H = 153
        LOW_S = 97
        HIGH_S = 255
        LOW_V = 45
        HIGH_V = 255

        # Thresholding the Images
        frame_HSV = cv.cvtColor(bg_removed, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, LOW_S, LOW_V), (high_H, HIGH_S, HIGH_V))

        # cv.imshow(window_capture_name, bg_removed)
        # images = np.hstack((color_image, depth_colormap))
        cv.imshow(window_detection_name, frame_threshold)

        # Contouring    
        contours, hierarchy = cv.findContours(frame_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        centroids = []
        areas = []
        MaxCentroid = [0,0]
        for cont in contours:
            currMoment = cv.moments(cont) # Get the Current Moment for the current cont
            conts_area = cv.contourArea(cont)
            try: 
                cx = int(currMoment['m10']/currMoment['m00'])
                cy = int(currMoment['m01']/currMoment['m00'])
                new_centroid = [cx, cy]
                centroids.append(new_centroid)
                areas.append(conts_area)
            except:
                pass
        try:
            # Find Largest Contour 
            MaxContTest = np.argmax(areas) # Want to get centroid of Max Contour
            MaxCentroid = centroids[MaxContTest]
            c_img_for_conts = cv.drawContours(c_img_for_conts, contours, -1, (255,204,0)) # if BGR (204,0,255)
            c_img_for_conts = cv.circle(c_img_for_conts, MaxCentroid, 10, (24,146,221), -1) # Thickness of -1 Fills in Circle 
        except: 
            print("404 Contour Not Found")

        # Depth Information Retrieval
        dpt_frame = aligned_depth_frame.as_depth_frame()
        pixel_distance_in_meters = dpt_frame.get_distance(MaxCentroid[0],MaxCentroid[1]) # Gives in Meters
        print(f"Distance (m): {pixel_distance_in_meters}\n")
        
        real_coords = rs.rs2_deproject_pixel_to_point(intr, [MaxCentroid[0],MaxCentroid[1]], pixel_distance_in_meters)
        print(f"Real Coords are: {real_coords[0], real_coords[1], real_coords[2]}")
        
        #  TODO: Image Filtering

        # Cartesian to Cylindrical
        radius_camera = np.sqrt(real_coords[0]**2 + real_coords[2]**2)

        cv.imshow("Contours", c_img_for_conts)
    
        # Phi Calculations
        depth_cam_to_arm = 0.344 # 0.35200000762939453 # 0.34200000762939453
        d_pen = pixel_distance_in_meters
        depth_pen_to_arm = d_pen - depth_cam_to_arm
        x_base_2_pen_robot_frame = 0.09066241 + 0.07264231145381927

        # real_x_to_pen = real_coords[0]
        phi = np.arctan(depth_pen_to_arm/x_base_2_pen_robot_frame)
        print(f"Phi is {phi}")
        
        RoboMoves.move(phi)
        # RoboMoves.ExtendArm(radius_camera=radius_camera)

        # bg_removed is our color_image with the background converted to gray.
        # Binary, Binary Inverted, Threshold Truncated, Threshold to Zero, Threshold to Zero Inverted
        # _, dst = cv.threshold(bg_removed, threshold_value, max_binary_value, threshold_type)

        # cv.namedWindow('Align Example', cv.WINDOW_NORMAL)
        # cv.imshow('Align Example', images)
        # cv.imshow('OG', bg_removed)
        # cv.imshow('Mask', mask)
        # cv.imshow('Res', res)
        key = cv.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv.destroyAllWindows()
            break
finally:
    pipeline.stop()