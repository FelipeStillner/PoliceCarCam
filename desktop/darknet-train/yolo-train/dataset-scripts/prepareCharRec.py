import cv2
import numpy as np
import os
import random
from os.path import splitext, basename
import re

# Function to parse the .txt file
def parse_car_txt(file_path):
    data = {}
    with open(file_path, 'r', encoding='iso-8859-1', errors='replace') as file:
        for line in file:
            key, value = line.strip().split(':', 1)
            data[key.strip()] = value.strip()

    corners = data['corners'].split()
    data['corners'] = [tuple(map(float, corner.split(','))) for corner in corners]

    if 'plate' in data:
        plate_text = data['plate'].replace(' ', '')
        if len(plate_text) == 7:
            data['letters'] = plate_text[:3]
            data['digits'] = plate_text[3:]
        else:
            print(f"Warning: Unexpected plate format in {file_path}")
            data['letters'], data['digits'] = '', ''
    
    return data

# Function to parse character bounding boxes
def parse_char_data(file_path):
    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            match = re.match(r"char \d+: (\d+) (\d+) (\d+) (\d+)", line.strip())
            if match:
                x, y, w, h = map(int, match.groups())
                boxes.append((x, y, w, h))
    return boxes

def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)

import cv2
import numpy as np
import os

def crop_plate(image_path, txt_path, output_folder_letters, output_folder_digits, padding=0):
    car_data = parse_car_txt(txt_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return

    char_boxes = parse_char_data(txt_path)

    if len(char_boxes) != 7:
        print(f"Warning: Unexpected number of characters in {txt_path}")
        return

    h_img, w_img, _ = image.shape  # Original image size

    for i, (x, y, w, h) in enumerate(char_boxes):
        # Apply padding before cropping
        x_min = max(x - padding, 0)
        y_min = max(y - padding, 0)
        x_max = min(x + w + padding, w_img)  # Keep within image width
        y_max = min(y + h + padding, h_img)  # Keep within image height

        print(f"Cropping character {i}: ({x_min}, {y_min}, {x_max}, {y_max})")  # Debugging

        char_img = image[y_min:y_max, x_min:x_max]  # Directly crop from original image

        if char_img.size == 0:
            print(f"Warning: Cropped character {i} is empty!")
            continue

        # Resize to fixed 32x96 size
        char_img_resized = cv2.resize(char_img, (32, 96))

        base_name = os.path.splitext(os.path.basename(image_path))[0] + f"_char{i}_pad{padding}"

        if i < 3:  # Letters
            class_label = ord(car_data['letters'][i]) - ord('A')
            output_img_path = os.path.join(output_folder_letters, base_name + ".png")
            output_label_path = os.path.join(output_folder_letters, base_name + ".txt")
        else:  # Digits
            class_label = int(car_data['digits'][i - 3])
            output_img_path = os.path.join(output_folder_digits, base_name + ".png")
            output_label_path = os.path.join(output_folder_digits, base_name + ".txt")

        # Compute the width and height of the cropped region
        cropped_w = x_max - x_min
        cropped_h = y_max - y_min

        # Normalize YOLO format values using cropped image dimensions
        w_norm = w / cropped_w
        h_norm = h / cropped_h
        x_c_norm = (x - x_min + w / 2) / cropped_w
        y_c_norm = (y - y_min + h / 2) / cropped_h

        # Save resized image
        cv2.imwrite(output_img_path, char_img_resized)

        # Show the cropped character for debugging
        # cv2.imshow("Cropped Character", char_img_resized)
        # cv2.waitKey(500)

        # Save YOLO label (normalized)
        with open(output_label_path, "w") as f:
            f.write(f"{class_label} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        print(f"Saved {output_img_path} with label {class_label}")

    cv2.destroyAllWindows()

# Main execution
output_folder_letters = '../charRec/full/val/'
output_folder_digits = '../digitRec/full/val/'
image_folder = '../../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/validation/'

image_subfolders = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
image_files = []
for folder in image_subfolders:
    image_files += sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Randomly select 1/7 of images
# image_files = random.sample(image_files, len(image_files) // 7)

# Apply different padding values
for padding in [ 0, 1, 2, 3, 4]:
    for image_path in image_files:
        txt_path = splitext(image_path)[0] + '.txt'
        crop_plate(image_path, txt_path, output_folder_letters, output_folder_digits, padding=padding)
