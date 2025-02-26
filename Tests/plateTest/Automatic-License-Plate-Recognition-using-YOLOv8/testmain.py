import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import time

# Set the device
# torch.device('cpu')
torch.cuda.set_device(0)  # Set to your desired GPU number
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device='cpu'
print(f'Using device: {device}')
print(torch.cuda.get_device_name())

# Initialize results dictionary and Sort tracker
results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')  # Vehicle detection model
# print(coco_model.device)

license_plate_detector = YOLO('license_plate_detector.pt')  # License plate detection model
# print(license_plate_detector.device)

# Path to folder containing images
# image_folder = '../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/testing/'
image_folder = '../yj4Iu2-UFPR-ALPR/UFPR-ALPR-dataset/testing/track0091'

# image_subfolder = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
# image_files = []
# for folder in image_subfolder:
#     sorted_=sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
#     image_files=image_files+sorted_
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

vehicles = [2, 3, 5, 7]  # Vehicle classes in COCO dataset (car, truck, bus, etc.)

# Process each image in the folder
for frame_nmr, image_path in enumerate(image_files):
    # print(image_path)
    # print(frame_nmr)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to read image: {image_path}")
        continue
    # cv2.imshow("thresh", frame)
    # cv2.waitKey(10)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    results[frame_nmr] = {}

    # Detect vehicles
    detections = coco_model.predict(source=frame,device=device,verbose=False)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_detector.predict(source=frame,device=device,verbose=False)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(1)
        # # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        # time.sleep(2)

        if car_id != -1:
            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # Process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            cv2.imshow("grayScale", license_plate_crop_gray)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
            cv2.waitKey(1)
            # time.sleep(2)
            blur_grey_plate = cv2.GaussianBlur(license_plate_crop_gray,(5,5),0)
            cv2.imshow("grayScale", license_plate_crop_gray)

            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 150, 255, cv2.THRESH_BINARY)
            # license_plate_crop_thresh = cv2.adaptiveThreshold(blur_grey_plate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #                                             cv2.THRESH_BINARY,11,2)
            # blur_grey_plate = cv2.GaussianBlur(license_plate_crop_gray,(5,5),0)
            # ret3,license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


            cv2.imshow("thresh", license_plate_crop_thresh)
            cv2.waitKey(10)
            # cv2.destroyAllWindows()
            img=license_plate_crop
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((1, 1, 1), np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)
            img = cv2.erode(img, kernel, iterations=1)
            cv2.imshow("original",license_plate_crop)
            # cv2.imshow("1", cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])

            cv2.imshow("2", cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
            cv2.imshow("5", cv2.adaptiveThreshold(cv2.bilateralFilter(img, 5, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2))

            cv2.waitKey(10)

            im2=cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            im5=cv2.adaptiveThreshold(cv2.bilateralFilter(img, 5, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            # Read license plate number
            # license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            # read_license_plate(im2)
            read_license_plate(im5)
            read_license_plate(img)

            time.sleep(1)

            # # print('macaco'+str(license_plate_text)+str(license_plate_text_score ))
            # if license_plate_text is not None:
            #     results[frame_nmr][car_id] = {
            #         'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
            #         'license_plate': {
            #             'bbox': [x1, y1, x2, y2],
            #             'text': license_plate_text,
            #             'bbox_score': score,
            #             'text_score': license_plate_text_score
            #         }
            #     }

# Write results to a CSV file
write_csv(results, './testMain.csv')
