
import torch


torch.cuda.set_device(0) # Set to your desired GPU number

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')#.to(device)
license_plate_detector = YOLO('license_plate_detector.pt')#.to(device)

# load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame,verbose=False)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame,verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            print(license_plate)

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                
                # blur_grey_plate=
                # license_plate_crop_thresh = cv2.adaptiveThreshold(blur_grey_plate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                #                                         cv2.THRESH_BINARY,11,2)
                img=license_plate_crop
                img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((1, 1), np.uint8)
                img = cv2.dilate(img, kernel, iterations=1)
                img = cv2.erode(img, kernel, iterations=1)
                cv2.imshow("original",license_plate_crop)
                # cv2.imshow("1", cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])

                cv2.imshow("2", cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])

                # cv2.imshow("3", cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])

                # cv2.imshow("4", cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2))

                # cv2.imshow("5", cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2))
                im6=cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
                cv2.imshow("6", cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2))

                cv2.imshow("7", license_plate_crop_thresh)
                cv2.waitKey(1)
                # cv2.destroyAllWindows()
                cv2.waitKey(1)
                time.sleep(2)
                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                read_license_plate(im6)

                # if license_plate_text is not None:
                #     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                #                                   'license_plate': {'bbox': [x1, y1, x2, y2],
                #                                                     'text': license_plate_text,
                #                                                     'bbox_score': score,
                #                                                     'text_score': license_plate_text_score}}

# write results
write_csv(results, './test3.csv')