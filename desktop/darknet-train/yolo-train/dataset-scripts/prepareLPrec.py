import cv2
import numpy as np
import os
from os.path import splitext, basename
import shutil

def parse_car_txt(file_path):
    """Parse the .txt file to extract the license plate corners."""
    data = {}
    with open(file_path, 'r', encoding='iso-8859-1', errors='replace') as file:
        for line in file:
            key, value = line.strip().split(':', 1)
            data[key.strip()] = value.strip()
    
    # Convert corners to a list of float tuples
    corners = data['corners'].split()
    data['corners'] = [tuple(map(float, corner.split(','))) for corner in corners]
    return data

def resize_image(image_path, output_path):
    """Resize the image to 416x416 and save it."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    resized_image = cv2.resize(image, (416, 448), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized_image)
    # print(f"Resized and saved: {output_path}")

def convert_to_yolo(corners, img_width, img_height):
    """Convert bounding box coordinates to YOLO format (class_id x_center y_center width height)."""
    x_min = min(c[0] for c in corners)
    x_max = max(c[0] for c in corners)
    y_min = min(c[1] for c in corners)
    y_max = max(c[1] for c in corners)
    
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_images(image_folder, output_folder):
    """Process all images and their respective labels."""
    image_files = []
    image_subfolders = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
    for folder in image_subfolders:
        sorted_images = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_files.extend(sorted_images)
    
    for image_path in image_files:
        txt_path = splitext(image_path)[0] + '.txt'
        label_output_path = os.path.join(output_folder, '', splitext(basename(image_path))[0] + '.txt')
        image_output_path = os.path.join(output_folder, '', splitext(basename(image_path))[0] + '.png')
        
        if os.path.exists(txt_path):
            car_data = parse_car_txt(txt_path)
            yolo_label = convert_to_yolo(car_data['corners'], 1920, 1080)
            with open(label_output_path, 'w') as f:
                f.write(yolo_label)
        
        resize_image(image_path, image_output_path)

# Paths
image_folder = '../../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/testing/'
output_folder = '../lpRec/test/'
os.makedirs(os.path.join(output_folder, ''), exist_ok=True)
os.makedirs(os.path.join(output_folder, ''), exist_ok=True)

# Process images
process_images(image_folder, output_folder)
