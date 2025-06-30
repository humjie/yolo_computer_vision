import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys
import argparse
import time
from ultralytics import YOLO
import supervision as sv

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--classes', help='Comma-separated list of classes to detect (example: "person,car,dog")',
                    default="shuttlecock")
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.0005)
parser.add_argument('--nms_thresh', help='Non-Maximum Suppression threshold (example: "0.5")',
                    default=0.4)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480")',
                    default="640x480")
parser.add_argument('--record', help='Record results and save it as "demo1.avi"',
                    action='store_true')
parser.add_argument('--max_det', help='Maximum number of detections per image (example: "50")',
                    default=50)

args = parser.parse_args()

# Parse user inputs
classes = args.classes.split(',')
min_thresh = float(args.thresh)
nms_thresh = float(args.nms_thresh)
max_det = int(args.max_det)
user_res = args.resolution
record = args.record

# Load YOLOWorld model
model = YOLO('yolov8s-worldv2.pt')

# Set classes for YOLOWorld
model.set_classes(classes)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
else:
    resW, resH = 640, 480

# Check if recording is valid and set up recording
if record:
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW*2, resH))

# Initialize RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, resW, resH, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, resW, resH, rs.format.z16, 30)
profile = pipe.start(cfg)

# Get color sensor for adjusting camera properties
color_sensor = profile.get_device().first_color_sensor()

# Initialize brightness setting
brightness_level = 0  # Default value
brightness_step = 1  # Increment value
try:
    # Get current brightness and range
    brightness_level = color_sensor.get_option(rs.option.brightness)
    brightness_range = color_sensor.get_option_range(rs.option.brightness)
    min_brightness = brightness_range.min
    max_brightness = brightness_range.max
    brightness_step = (max_brightness - min_brightness) / 20  # Divide range into 20 steps
except Exception as e:
    print(f"Warning: Could not get brightness range: {e}")
    min_brightness = -64
    max_brightness = 64
    brightness_step = 4

# Initialize contrast setting
contrast_level = 0  # Default value
contrast_step = 1  # Increment value
try:
    # Get current contrast and range
    contrast_level = color_sensor.get_option(rs.option.contrast)
    contrast_range = color_sensor.get_option_range(rs.option.contrast)
    min_contrast = contrast_range.min
    max_contrast = contrast_range.max
    contrast_step = (max_contrast - min_contrast) / 20  # Divide range into 20 steps
except Exception as e:
    print(f"Warning: Could not get contrast range: {e}")
    min_contrast = 0
    max_contrast = 100
    contrast_step = 5

# Initialize exposure setting
exposure_level = 0  # Default value
exposure_step = 1  # Increment value
auto_exposure = True  # Start with auto exposure
try:
    # Get current exposure and range
    auto_exposure = color_sensor.get_option(rs.option.enable_auto_exposure) == 1.0
    exposure_level = color_sensor.get_option(rs.option.exposure)
    exposure_range = color_sensor.get_option_range(rs.option.exposure)
    min_exposure = exposure_range.min
    max_exposure = exposure_range.max
    exposure_step = (max_exposure - min_exposure) / 100  # Divide range into 20 steps
except Exception as e:
    print(f"Warning: Could not get exposure range: {e}")
    min_exposure = 1
    max_exposure = 10000
    exposure_step = 500

# Create alignment primitive with color as its target stream
align = rs.align(rs.stream.color)

# Create colorizer for depth visualization
colorizer = rs.colorizer()

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
    pipe.wait_for_frames()

# Create windows for displaying the images
cv2.namedWindow('YOLOWorld Detection with RealSense', cv2.WINDOW_AUTOSIZE)

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 30

# Create supervision annotators with improved settings
box_annotator = sv.BoxAnnotator(
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    text_thickness=1,
    text_scale=0.5,
    text_padding=5
)

# Performance optimization: skip frames for inference

# Performance optimization: skip frames for inference
inference_skip = 2  # Process every 2nd frame
frame_count = 0
last_detections = sv.Detections.empty()

while True:
    t_start = time.perf_counter()
    
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

    # Performance optimization: run inference every N frames
    frame_count += 1
    if frame_count % inference_skip == 0:
        # Run YOLOWorld inference on the color image with optimized parameters
        results = model(
            color_image, 
            conf=min_thresh,
            iou=nms_thresh,
            max_det=max_det,
            verbose=False  # Reduce output noise
        )
        
        # Convert to supervision detections and apply advanced NMS
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Apply additional filtering
        detections = detections.with_nms(threshold=nms_thresh)
        
        # Filter by confidence again after NMS
        detections = detections[detections.confidence > min_thresh]
        
        # Sort by confidence (highest first)
        sorted_indices = np.argsort(detections.confidence)[::-1]
        detections = detections[sorted_indices]
        
        # Limit maximum detections
        if len(detections) > max_det:
            detections = detections[:max_det]
            
        last_detections = detections
    else:
        # Use previous frame's detections for performance
        detections = last_detections

    # Initialize variable for basic object counting example
    object_count = len(detections)

    # Make a copy of the color image for drawing detections
    detection_image = color_image.copy()

    # Prepare labels with depth information
    labels = []
    unique_objects = {}  # Track unique objects to avoid duplicates
    
    for i in range(len(detections)):
        # Get bounding box coordinates
        bbox = detections.xyxy[i].astype(int)
        xmin, ymin, xmax, ymax = bbox
        
        # Calculate the distance to the object's center
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        
        # Get the depth value at the center of the bounding box (in meters)
        depth_value = aligned_depth_frame.get_distance(center_x, center_y)
        
        # Get class information
        class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        class_name = classes[class_id] if class_id < len(classes) else "unknown"
        confidence = detections.confidence[i] if detections.confidence is not None else 0.0
        
        # Create unique identifier for object (class + approximate position)
        obj_key = f"{class_name}_{center_x//50}_{center_y//50}"
        
        # Check for duplicate objects in similar positions
        if obj_key in unique_objects:
            # Keep the one with higher confidence
            if confidence > unique_objects[obj_key]['confidence']:
                unique_objects[obj_key] = {
                    'confidence': confidence,
                    'depth': depth_value,
                    'index': i
                }
        else:
            unique_objects[obj_key] = {
                'confidence': confidence,
                'depth': depth_value,
                'index': i
            }    # Create labels only for unique objects
    valid_indices = []
    
    # Create labels for each unique object
    for obj_key, obj_data in unique_objects.items():
        i = obj_data['index']
        valid_indices.append(i)
        
        class_id = int(detections.class_id[i])
        class_name = classes[class_id] if class_id < len(classes) else "unknown"
        confidence = obj_data['confidence']
        depth_value = obj_data['depth']
        
        # Create label with depth information
        if depth_value > 0 and depth_value < 10:  # Filter unrealistic depths
            label = f"{class_name}: {confidence:.2f} | {depth_value:.2f}m"
        else:
            label = f"{class_name}: {confidence:.2f}"
        
        labels.append(label)
    
    # Filter detections to only show unique objects
    if valid_indices:
        valid_indices = np.array(valid_indices)
        filtered_detections = detections[valid_indices]
        object_count = len(filtered_detections)
        
        # Annotate image using supervision
        detection_image = box_annotator.annotate(scene=detection_image, detections=filtered_detections)
        detection_image = label_annotator.annotate(scene=detection_image, detections=filtered_detections, labels=labels)

    # Stack both images horizontally for display
    display_image = np.hstack((detection_image, depth_colormap))
    
    # Calculate and display frame rate
    cv2.putText(display_image, f'FPS: {avg_frame_rate:0.2f}', (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)    # Display detection results
    cv2.putText(display_image, f'Objects: {object_count}', (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display camera settings
    y_offset = 80  # Starting y position for settings text
    y_offset = 80  # Starting y position for settings text
    
    # Display brightness setting
    cv2.putText(display_image, f'Brightness: {brightness_level:.1f} [+/-]', (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset += 30
    
    # Display contrast setting
    cv2.putText(display_image, f'Contrast: {contrast_level:.1f} [c/C]', (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset += 30
    
    # Display exposure setting
    if auto_exposure:
        cv2.putText(display_image, f'Exposure: AUTO [a to toggle, [ ] to adjust]', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(display_image, f'Exposure: {exposure_level:.1f} [a to toggle, [ ] to adjust]', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Display the images
    cv2.imshow('YOLOWorld Detection with RealSense', display_image)
    
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
    elif key & 0xFF == ord('r'):  # Press 'r' to reset inference
        frame_count = 0
    elif key & 0xFF == ord('+') or key == ord('='):  # Press '+' to increase brightness
        # Increase brightness (ensure we don't exceed max)
        new_brightness = min(brightness_level + brightness_step, max_brightness)
        if new_brightness != brightness_level:
            try:
                color_sensor.set_option(rs.option.brightness, new_brightness)
                brightness_level = new_brightness
                print(f"Brightness increased to: {brightness_level:.1f}")
            except Exception as e:
                print(f"Error adjusting brightness: {e}")
    elif key & 0xFF == ord('-') or key == ord('_'):  # Press '-' to decrease brightness
        # Decrease brightness (ensure we don't go below min)
        new_brightness = max(brightness_level - brightness_step, min_brightness)
        if new_brightness != brightness_level:
            try:
                color_sensor.set_option(rs.option.brightness, new_brightness)
                brightness_level = new_brightness
                print(f"Brightness decreased to: {brightness_level:.1f}")
            except Exception as e:
                print(f"Error adjusting brightness: {e}")
    elif key & 0xFF == ord('c'):  # Press 'c' to decrease contrast
        # Decrease contrast (ensure we don't go below min)
        new_contrast = max(contrast_level - contrast_step, min_contrast)
        if new_contrast != contrast_level:
            try:
                color_sensor.set_option(rs.option.contrast, new_contrast)
                contrast_level = new_contrast
                print(f"Contrast decreased to: {contrast_level:.1f}")
            except Exception as e:
                print(f"Error adjusting contrast: {e}")
    elif key & 0xFF == ord('C'):  # Press 'C' to increase contrast
        # Increase contrast (ensure we don't exceed max)
        new_contrast = min(contrast_level + contrast_step, max_contrast)
        if new_contrast != contrast_level:
            try:
                color_sensor.set_option(rs.option.contrast, new_contrast)
                contrast_level = new_contrast
                print(f"Contrast increased to: {contrast_level:.1f}")
            except Exception as e:
                print(f"Error adjusting contrast: {e}")
    elif key & 0xFF == ord('['):  # Press '[' to decrease exposure
        # First disable auto exposure if enabled
        if auto_exposure:
            try:
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                auto_exposure = False
                print("Auto exposure disabled")
            except Exception as e:
                print(f"Error disabling auto exposure: {e}")
        
        # Decrease exposure (ensure we don't go below min)
        new_exposure = max(exposure_level - exposure_step, min_exposure)
        if new_exposure != exposure_level:
            try:
                color_sensor.set_option(rs.option.exposure, new_exposure)
                exposure_level = new_exposure
                print(f"Exposure decreased to: {exposure_level:.1f}")
            except Exception as e:
                print(f"Error adjusting exposure: {e}")
    elif key & 0xFF == ord(']'):  # Press ']' to increase exposure
        # First disable auto exposure if enabled
        if auto_exposure:
            try:
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                auto_exposure = False
                print("Auto exposure disabled")
            except Exception as e:
                print(f"Error disabling auto exposure: {e}")
        
        # Increase exposure (ensure we don't exceed max)
        new_exposure = min(exposure_level + exposure_step, max_exposure)
        if new_exposure != exposure_level:
            try:
                color_sensor.set_option(rs.option.exposure, new_exposure)
                exposure_level = new_exposure
                print(f"Exposure increased to: {exposure_level:.1f}")
            except Exception as e:
                print(f"Error adjusting exposure: {e}")
    elif key & 0xFF == ord('a'):  # Press 'a' to toggle auto exposure
        try:
            auto_exposure = not auto_exposure
            color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
            print(f"Auto exposure {'enabled' if auto_exposure else 'disabled'}")
            if not auto_exposure:
                # Update exposure level display when switching to manual
                exposure_level = color_sensor.get_option(rs.option.exposure)
        except Exception as e:
            print(f"Error toggling auto exposure: {e}")
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer
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