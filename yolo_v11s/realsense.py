import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import os
import sys
import argparse
import glob
import time
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480")',
                    default="640x480")
parser.add_argument('--record', help='Record results and save it as "demo1.avi"',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labelmap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
else:
    resW, resH = 640, 480

# Check if recording is valid and set up recording
if record:
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW*2, resH))

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, resW, resH, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, resW, resH, rs.format.z16, 30)
profile = pipe.start(cfg)

# Create alignment primitive with color as its target stream:
align = rs.align(rs.stream.color)

# Create colorizer for depth visualization
colorizer = rs.colorizer()

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
    pipe.wait_for_frames()
  
# Create windows for displaying the images
cv2.namedWindow('YOLO Detection with RealSense', cv2.WINDOW_AUTOSIZE)

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 30

while True:
    t_start = time.perf_counter()
    
    # Store next frameset for later processing:
    # Wait for a coherent pair of frames: depth and color
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    # Process aligned frames
    aligned_frames = align.process(frameset)
    aligned_color_frame = aligned_frames.get_color_frame()
    aligned_depth_frame = aligned_frames.get_depth_frame()
  
    # Convert aligned frames to numpy arrays
    color_image = np.asanyarray(aligned_color_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
        np.asanyarray(aligned_depth_frame.get_data()), alpha=0.03), 
        cv2.COLORMAP_JET)

    # Resize frame to desired display resolution if needed
    if resize and (color_image.shape[1] != resW or color_image.shape[0] != resH):
        color_image = cv2.resize(color_image, (resW, resH))
        depth_colormap = cv2.resize(depth_colormap, (resW, resH))

    # Run YOLO inference on the color image
    results = model(color_image, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Make a copy of the color image for drawing detections
    detection_image = color_image.copy()

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(detection_image, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(detection_image, (xmin, label_ymin-labelSize[1]-10), 
                         (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(detection_image, label, (xmin, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Calculate the distance to the object's center
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            
            # Get the depth value at the center of the bounding box (in meters)
            depth_value = aligned_depth_frame.get_distance(center_x, center_y)
            
            # Display the depth value if available (not 0)
            if depth_value > 0:
                depth_text = f'Depth: {depth_value:.2f}m'
                cv2.putText(detection_image, depth_text, 
                           (xmin, label_ymin+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Basic example: count the number of objects in the image
            object_count += 1

    # Stack both images horizontally for display
    display_image = np.hstack((detection_image, depth_colormap))
    
    # Calculate and display frame rate
    cv2.putText(display_image, f'FPS: {avg_frame_rate:0.2f}', (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display detection results
    cv2.putText(display_image, f'Objects: {object_count}', (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display the images
    cv2.imshow('YOLO Detection with RealSense', display_image)
    
    # Record if enabled
    if record:
        recorder.write(display_image)
    
    # Handle key presses
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # Press 'q' to quit
        break
    elif key & 0xFF == ord('s'):  # Press 's' to pause 
        cv2.waitKey(0)
    elif key & 0xFF == ord('p'):  # Press 'p' to save a picture
        cv2.imwrite('capture.png', display_image)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if record:
    recorder.release()
pipe.stop()
cv2.destroyAllWindows()