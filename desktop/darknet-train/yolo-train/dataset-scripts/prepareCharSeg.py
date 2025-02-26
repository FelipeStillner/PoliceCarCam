import cv2
import numpy as np
import os
from os.path import splitext, basename
import re

def parse_car_txt(file_path):
    data = {}
    with open(file_path, 'r', encoding='iso-8859-1', errors='replace') as file:
        for line in file:
            key, value = line.strip().split(':', 1)
            data[key.strip()] = value.strip()
    
    corners = data['corners'].split()
    data['corners'] = [tuple(map(float, corner.split(','))) for corner in corners]
    return data

def parse_char_data(file_path):
    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            match = re.match(r"char \d+: (\d+) (\d+) (\d+) (\d+)", line.strip())
            if match:
                x, y, w, h = map(int, match.groups())
                boxes.append((x, y, w, h))
    return boxes

def crop_plate(image_path, txt_path, output_img_path, output_label_path):
    car_data = parse_car_txt(txt_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return
    
    corners = np.array(car_data['corners'], dtype=np.float32)
    x_min, y_min = corners[:, 0].min(), corners[:, 1].min()
    x_max, y_max = corners[:, 0].max(), corners[:, 1].max()
    width, height = int(x_max - x_min), int(y_max - y_min)
    
    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    resized_image = cv2.resize(cropped_image, (256, 96), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_img_path, resized_image)
    # Compute the width and height of the cropped region
    cropped_w = x_max - x_min
    cropped_h = y_max - y_min

    char_boxes = parse_char_data(txt_path)
    new_labels = []
    for (x, y, w, h) in char_boxes:
        # x_rel = (x - x_min) / width
        # y_rel = (y - y_min) / height
        # w_rel = w / width
        # h_rel = h / height

        # Normalize YOLO format values using cropped image dimensions
        w_norm = w / cropped_w
        h_norm = h / cropped_h
        x_c_norm = (x - x_min + w / 2) / cropped_w
        y_c_norm = (y - y_min + h / 2) / cropped_h

        new_labels.append(f"0 {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    with open(output_label_path, "w") as f:
        f.write("\n".join(new_labels))
    
    # print(f"Cropped image saved as: {output_img_path}")
    # print(f"Updated labels saved as: {output_label_path}")

output_path_base = '../characterSeg/dataset/val/'
image_folder = '../../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/validation/'

image_subfolders = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
image_files = []
for folder in image_subfolders:
    image_files += sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for image_path in image_files:
    txt_path = splitext(image_path)[0] + '.txt'
    output_img_path = output_path_base + splitext(basename(image_path))[0] + '.png'
    output_label_path = output_path_base + splitext(basename(image_path))[0] + '.txt'
    crop_plate(image_path, txt_path, output_img_path, output_label_path)
