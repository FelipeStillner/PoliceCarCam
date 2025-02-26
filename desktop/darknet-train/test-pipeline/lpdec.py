import torch
# Set the device
# torch.device('cpu')
torch.cuda.set_device(0)  # Set to your desired GPU number
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device='cpu'
print(f'Using device: {device}')
print(torch.cuda.get_device_name())

import cv2
import numpy as np
# import python.darknet as darknet

import os
import cv2
import numpy as np
from ultralytics import YOLO
# from sort.sort import *
# from util import get_car, read_license_plate, write_csv
import time
from ultralytics import YOLO
import cv2
import os
import glob

# Load YOLO model (replace with your specific model if needed)
model = YOLO('license_plate_detector.pt')  # Use the pretrained YOLOv8 model

# Directory containing the images
image_dir =  '../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/validation/'

output_dir = 'output_crops'   # Output directory for cropped license plates
os.makedirs(output_dir, exist_ok=True)

# Function to crop and save license plates
def crop_license_plate(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Perform inference
    results = model(img)[0].boxes.data.tolist()

    # Iterate through detected objects
    i=0
    for license_plate in results:  # For each detection
        x1, y1, x2, y2, score, class_id = license_plate
        conf = score # Confidence score
        if conf > 0.5:  # Only consider detections with a confidence score above 0.5
            # x1, y1, x2, y2 = map(int, det[:4])  # Bounding box coordinates

            # Crop the license plate from the image
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

            # Save the cropped image
            output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_crop_{i}.jpg")
            cv2.imwrite(output_path, cropped_img)

            print(f"Cropped and saved: {output_path}")
        i+=1

# Get all image paths in the directory
image_folder = '../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/validation/'

image_subfolders = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
image_files = []
for folder in image_subfolders:
    image_files += sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Process images in batches (let's say a batch size of 16)
# Process all images in the directory
for filename in image_files:
    # image_path = os.path.join(image_dir, filename)
    crop_license_plate(filename)
