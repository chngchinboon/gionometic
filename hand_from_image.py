# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import time
import pyrealsense2 as rs

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        print(dir_path + '/openpose/build/python/openpose/Release')
        # sys.path.append(dir_path + '/../../python/openpose/Release');
        sys.path.append(dir_path + '/openpose/build/python/openpose/Release');
        # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose/build/x64/Release;' +  dir_path + '/openpose/build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "openpose/models/"
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# RealSense Depth Camera Setup
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)

# Get data scale from the device and convert to meters
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# webcame setup
# cap = cv2.VideoCapture(0)
# hasFrame, frame = cap.read()

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    while 1:
        # Read image and face rectangle locations
        # imageToProcess = cv2.imread(args[0].image_path)
        # hasFrame, imageToProcess = cap.read()
        #
        # if not hasFrame:
        #     cv2.waitKey()
        #     break
        frameset = pipe.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frameset)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

        r1x=300
        r1y=100
        r2w=300
        r2h=300

        handRectangles = [
            # Left/Right hands person 0
            [
            op.Rectangle(0., 0., 0., 0.),
            op.Rectangle(r1x, r1y, r2w, r2h),
            ],
            # Left/Right hands person 1
            # [
            # op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
            # op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
            # ],
            # # Left/Right hands person 2
            # [
            # op.Rectangle(185.692673, 303.112244, 157.587555, 157.587555),
            # op.Rectangle(88.984360, 268.866547, 117.818230, 117.818230),
            # ]
        ]

        # Create new datum
        datum = op.Datum()
        datum.cvInputData = color_image
        datum.handRectangles = handRectangles

        # Process and display image
        opWrapper.emplaceAndPop([datum])
        print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        frame = datum.cvOutputData

        cv2.rectangle(frame, (r1x, r1y), (r1x+r2w, r1y+r2h), (0, 255, 0), 1, 1)

        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        # cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
