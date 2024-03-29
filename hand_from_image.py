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
        os.environ['PATH'] = os.environ[
                                 'PATH'] + ';' + dir_path + '/openpose/build/x64/Release;' + dir_path + '/openpose/build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg",
                    help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "openpose/models/"
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0

# openpose finger segments
keypoint_segments = ((0, 1, 'thumb base'),  # 0
                     (1, 2, 'thumb proximal phalanx'),  # 1
                     (2, 3, 'thumb middle phalanx'),  # 2
                     (3, 4, 'thumb distal phalanx'),  # 3
                     (0, 5, 'index base'),  # 4
                     (5, 6, 'index proximal phalanx'),  # 5
                     (6, 7, 'index middle phalanx'),  # 6
                     (7, 8, 'index distal phalanx'),  # 7
                     (0, 9, 'middle base'),  # 8
                     (9, 10, 'middle proximal phalanx'),  # 9
                     (10, 11, 'middle middle phalanx'),  # 10
                     (11, 12, 'middle distal phalanx'),  # 11
                     (0, 13, 'ring base'),  # 12
                     (13, 14, 'ring proximal phalanx'),  # 13
                     (14, 16, 'ring middle phalanx'),  # 14
                     (15, 16, 'ring distal phalanx'),  # 15
                     (0, 17, 'baby base'),  # 16
                     (17, 18, 'baby proximal phalanx'),  # 17
                     (18, 19, 'baby middle phalanx'),  # 18
                     (19, 20, 'baby distal phalanx'),  # 19
                     )

keypoint_pairs = ((0, 1, 2, '0_1_2','thumb CMC'),
                  (1, 2, 3, '1_2_3','thumb MCP'),
                  (2, 3, 4, '2_3_4', 'thumb IP'),
                  (0, 5, 6, '0_5_6', 'index PIP'),
                  (5, 6, 7, '5_6_7', 'index MCP'),
                  (6, 7, 8, '6_7_8', 'index DIP'),
                  (0, 9, 10, '0_9_10', 'middle PIP'),
                  (9, 10, 11, ' 9_10_11', 'middle MCP'),
                  (10, 11, 12, '10_11_12', 'middle DIP'),
                  (0, 13, 14, '0_13_14', 'ring PIP'),
                  (13, 14, 15, '13_14_15', 'ring MCP'),
                  (14, 15, 16, '14_15_16', 'ring DIP'),
                  (0, 17, 18, '0_17_18', 'baby PIP'),
                  (17, 18, 19, '17_18_19', 'baby MCP'),
                  (18, 19, 20, '18_19_20', 'baby DIP')
                  )


def xyd_to_xyz(pt, aligned_depth_frame, rs, depth_intrin):
    d_px = [pt[0], pt[1]]
    d_value = aligned_depth_frame.get_distance(*d_px)
    d_pt = np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, d_px, d_value))
    return d_pt


def process_kp_angles(kpdata, aligned_depth_frame, rs, depth_intrin, thresh):
    kp_angle = {key[3]: 0 for key in keypoint_pairs}
    # print('#################################')
    for pair in keypoint_pairs:
        if (kpdata[pair[0]][2] < thresh) | (kpdata[pair[1]][2] < thresh) | (
                kpdata[pair[2]][2] < thresh):  # skip showing points with low confidence
            # print(f'angle for {pair[3]}: Bad')
            kp_angle[pair[3]] = np.NaN
            continue
        # check confidence of point, if poor, skip
        pt0 = xyd_to_xyz(kpdata[pair[0]], aligned_depth_frame, rs, depth_intrin)
        pt1 = xyd_to_xyz(kpdata[pair[1]], aligned_depth_frame, rs, depth_intrin)
        pt2 = xyd_to_xyz(kpdata[pair[2]], aligned_depth_frame, rs, depth_intrin)

        start_vector = pt1 - pt0
        norm_start_vector = np.linalg.norm(start_vector)
        end_vector = pt2 - pt1
        norm_end_vector = np.linalg.norm(end_vector)
        angle = np.dot(start_vector, end_vector) / (norm_start_vector * norm_end_vector)
        # print(f'angle for {pair[3]}: {np.rad2deg(angle):.2f}')
        kp_angle[pair[3]] = angle
    return kp_angle


def store_maxmin(kp_maxmin, kp_angles):
    new_maxmin = kp_maxmin.copy()
    for key, angle in kp_angles.items():
        new_maxmin[key] = np.array((np.nanmax((kp_maxmin[key][0], angle)),
                                    np.nanmin((kp_maxmin[key][1], angle))))
    return new_maxmin


# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-', '')
    if key not in params:
        params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-', '')
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
fframe = 1
kp_maxmin = {key[3]: np.array((0, 0)) for key in keypoint_pairs}
confidence_threshold = 0.1

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    while 1:
        ts = time.perf_counter()
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

        if fframe == 1:
            first_frame = color_image.copy()
            fframe = 0

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        # color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        # print(f'width: {color_intrin.width} height: {color_intrin.height}')
        # depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # default image from RS: w848 h480. seems like max resolu input for openpose: 300x300
        r1x = 300
        r1y = 100
        r2w = 300  # 300
        r2h = 300  # 300

        handRectangles = [
            # Left/Right hands person 0
            [
                op.Rectangle(0., 0., 0., 0.),  # disable left hand
                op.Rectangle(r1x, r1y, r2w, r2h),
            ],

        ]

        # Create new datum
        datum = op.Datum()
        datum.cvInputData = color_image
        datum.handRectangles = handRectangles  # need to modify here with a hand locator.

        # Process and display image
        opWrapper.emplaceAndPop([datum])
        # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        frame = datum.cvOutputData
        kpdata = datum.handKeypoints[1][0]

        cv2.rectangle(frame, (r1x, r1y), (r1x + r2w, r1y + r2h), (0, 255, 0), 1, 1)
        kp_angles = process_kp_angles(kpdata, aligned_depth_frame, rs, depth_intrin, confidence_threshold)  # get angles between segments
        # store max min
        kp_maxmin = store_maxmin(kp_maxmin, kp_angles)
        # print(kp_maxmin)
        for idx, kp in enumerate(kpdata):
            if kp[2] < confidence_threshold:  # skip showing points with low confidence
                continue
            # print(f'p{idx}: {kp}')
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(idx), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, .8,
                        (0, 0, 255), 2, lineType=cv2.LINE_AA)

        table_start = (0, 0)
        cell_height = 25
        cell_width = 50
        frame = cv2.resize(frame, (1280, 720))

        i = 0
        for k, v in kp_maxmin.items():  # table of values
            i += 1
            cv2.putText(frame, f"{k}: Max: {np.rad2deg(v[0]):3.2f}, Min: {np.rad2deg(v[1]):3.2f}, Avg: {np.rad2deg(v[0])-np.rad2deg(v[1])/2:3.2f}",
                        (table_start[0], table_start[1] + cell_height * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        te = time.perf_counter() - ts
        fps = 1 / te

        cv2.putText(frame, f"{fps:.2f} FPS", (0, frame.shape[0] - cell_height), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    (0, 0, 255), 2, lineType=cv2.LINE_AA)

        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", frame)

        # print(f'Time elasped: {te:.2f}, FPS: {fps:.2f}')
        key = cv2.waitKey(1)
        if key == 27:
            break
        # cv2.waitKey(0)
except Exception as e:
    print(e)
    # sys.exit(-1)

for k,v in kp_maxmin.items():
    print(f'{[v[4] for v in keypoint_pairs if k in v][0]}: Max: {np.rad2deg(v[0]):3.2f}, Min: {np.rad2deg(v[1]):3.2f}, '
          f'Avg: {np.rad2deg(v[0])-np.rad2deg(v[1])/2:3.2f}')
