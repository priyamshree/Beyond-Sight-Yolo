import cv2
import numpy as np
import os

# Threshold settings
thres = 0.45  # Confidence threshold
nms_threshold = 0.5  # Non-Maximum Suppression threshold

# Reference object parameters (Change these values based on your setup)
KNOWN_WIDTH = 15  # Width of the object in cm
KNOWN_DISTANCE = 50  # Distance of the object from the camera in cm
FOCAL_LENGTH = 700  # Approximate focal length (needs calibration)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height
cap.set(10, 150)  # Brightness

# Load class names
classNames = []
classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model configuration and weights
configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))

# Load DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def calculate_depth(known_width, focal_length, perceived_width):
    if perceived_width > 0:
        return (known_width * focal_length) / perceived_width
    return -1  # Invalid depth

while True:
    # Capture frame
    success, image = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    # Detect objects
    classIds, confs, bbox = net.detect(image, confThreshold=thres)
    
    if isinstance(classIds, tuple):  # If no detections, OpenCV returns empty tuple
        classIds = np.array([])
        confs = np.array([])
        bbox = []
    
    classIds = classIds.flatten().astype(int) if classIds.size > 0 else []
    confs = confs.flatten() if confs.size > 0 else []
    bbox = list(bbox)
    
    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    # Draw bounding boxes and depth estimation
    if len(indices) > 0:
        for i in indices.flatten():
            if 0 <= i < len(classIds):
                x, y, w, h = bbox[i]
                class_index = classIds[i] - 1

                # Ensure class_index is within bounds
                label = classNames[class_index] if 0 <= class_index < len(classNames) else "Unknown"

                # Estimate depth
                depth = calculate_depth(KNOWN_WIDTH, FOCAL_LENGTH, w)
                depth_text = f"Depth: {depth:.2f} cm" if depth > 0 else "Depth: Unknown"

                # Draw rectangle, label, and depth
                cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.putText(image, label, (x + 10, y + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, depth_text, (x + 10, y + 60),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    # Display output
    cv2.imshow("Depth Estimation", image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
