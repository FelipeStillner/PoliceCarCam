import os
import numpy as np
import cv2
from rknnlite.api import RKNNLite

from neural_network.utils import yolov3_post_process, yolov3_post_process_char_seg
import neural_network.config_charRec as config_charRec
import neural_network.config_charSeg as config_charSeg
import neural_network.config_digitRec as config_digitRec
import neural_network.config_lpRec as config_lpRec


class LicensePlateRecognizer:
    def __init__(self):
        # Initialize and load the RKNN models only once
        self.rknn_plate = RKNNLite(verbose=False)
        if self.rknn_plate.load_rknn(config_lpRec.RKNN_MODEL_PATH) != 0:
            print('Load RKNN model failed for plate detection!')
        if self.rknn_plate.init_runtime() != 0:
            print('Init runtime environment failed for plate detection!')

        self.rknn_char_seg = RKNNLite(verbose=False)
        if self.rknn_char_seg.load_rknn(config_charSeg.RKNN_MODEL_PATH) != 0:
            print('Load RKNN model failed for character segmentation!')
        if self.rknn_char_seg.init_runtime() != 0:
            print('Init runtime environment failed for character segmentation!')

        self.rknn_digit = RKNNLite(verbose=False)
        if self.rknn_digit.load_rknn(config_digitRec.RKNN_MODEL_PATH) != 0:
            print('Load RKNN model failed for digit detection!')
        if self.rknn_digit.init_runtime() != 0:
            print('Init runtime environment failed for digit detection!')

        self.rknn_letter = RKNNLite(verbose=False)
        if self.rknn_letter.load_rknn(config_charRec.RKNN_MODEL_PATH) != 0:
            print('Load RKNN model failed for letter detection!')
        if self.rknn_letter.init_runtime() != 0:
            print('Init runtime environment failed for letter detection!')

    def detect_plate(self, image):
        """Detect license plate(s) using the preloaded plate detection model."""
        img = cv2.resize(image, (416, 448))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0)
        img = img.astype(np.uint8)

        outputs = self.rknn_plate.inference(inputs=[img_input])

        input_data = []
        for output in outputs:
            batch, channels, height, width = output.shape
            SPAN = channels // config_lpRec.LISTSIZE
            output = output.reshape(batch, SPAN, config_lpRec.LISTSIZE, height, width)
            output = output.transpose(0, 3, 4, 1, 2)
            input_data.append(output[0])

        boxes, classes, scores = yolov3_post_process(
            input_data,
            config_lpRec.anchors,
            config_lpRec.masks,
            config_lpRec.OBJ_THRESH,
            config_lpRec.NMS_THRESH
        )
        if boxes is None:
            return [], []

        # Pair boxes with scores and sort by descending score (best first)
        detections = list(zip(boxes, scores))
        sorted_detections = sorted(detections, key=lambda x: float(x[1]), reverse=True)

        detected_plates = []
        sorted_scores = []
        for box, score in sorted_detections:
            x, y, w, h = box
            x *= image.shape[1]
            y *= image.shape[0]
            w *= image.shape[1]
            h *= image.shape[0]

            # Expand box by 20% (10% on each side)
            expand_x = int(w * 0.1)
            expand_y = int(h * 0.1)

            left = max(0, int(x - expand_x))
            top = max(0, int(y - expand_y))
            right = min(image.shape[1], int(x + w + expand_x))
            bottom = min(image.shape[0], int(y + h + expand_y))

            detected_plates.append(image[top:bottom, left:right])
            sorted_scores.append(score)

        return detected_plates, sorted_scores

    def segment_chars(self, image):
        """
        Segment characters from the input image.
        Returns up to 7 highest-scoring character detections,
        ordered left-to-right.
        """
        img = cv2.resize(image, (256, 96))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0)

        outputs = self.rknn_char_seg.inference(inputs=[img_input])

        input_data = []
        for output in outputs:
            batch, channels, height, width = output.shape
            SPAN = channels // config_charSeg.LISTSIZE
            output = output.reshape(batch, SPAN, config_charSeg.LISTSIZE, height, width)
            output = output.transpose(0, 3, 4, 1, 2)
            input_data.append(output[0])

        boxes, classes, scores = yolov3_post_process_char_seg(
            input_data,
            config_charSeg.anchors,
            config_charSeg.masks,
            config_charSeg.OBJ_THRESH,
            config_charSeg.NMS_THRESH
        )
        if boxes is None:
            print("No characters detected")
            return []

        detections = list(zip(boxes, scores))
        # Get the top 7 highest-scoring detections
        top_detections = sorted(detections, key=lambda x: float(x[1]), reverse=True)[:7]
        # Order left-to-right based on the x-coordinate of the box
        top_detections = sorted(top_detections, key=lambda x: float(x[0][0]))

        cropped_chars = []
        for box, _ in top_detections:
            x, y, w, h = box
            x *= image.shape[1]
            y *= image.shape[0]
            w *= image.shape[1]
            h *= image.shape[0]

            left = max(0, int(x))
            top = max(0, int(y))
            right = min(image.shape[1], int(x + w))
            bottom = min(image.shape[0], int(y + h))

            char_img = image[top:bottom, left:right]
            # Resize to target dimensions (32x96)
            char_img = cv2.resize(char_img, (32, 96))
            cropped_chars.append(char_img)

        return cropped_chars

    def detect_digit(self, img):
        """
        Detect digit in an image using the preloaded digit recognition model.
        Returns the detected digit and its confidence.
        """
        input_image_width = config_digitRec.input_w
        input_image_height = config_digitRec.input_h

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_image_width, input_image_height))
        img_resized = img_resized.astype(np.uint8)
        img_input = np.expand_dims(img_resized, axis=0)

        outputs = self.rknn_digit.inference(inputs=[img_input])

        input_data = []
        LISTSIZE = 4 + 1 + config_digitRec.NUM_CLS
        for output in outputs:
            batch, channels, height, width = output.shape
            SPAN = channels // LISTSIZE
            output = output.reshape(batch, SPAN, LISTSIZE, height, width)
            output = output.transpose(0, 3, 4, 1, 2)
            input_data.append(output[0])

        boxes, classes, scores = yolov3_post_process(
            input_data,
            config_digitRec.anchors,
            config_digitRec.masks,
            config_digitRec.OBJ_THRESH,
            config_digitRec.NMS_THRESH
        )

        if boxes is not None and len(boxes) > 0:
            highest_confidence_idx = np.argmax(scores)
            detected_digit = config_digitRec.CLASSES[classes[highest_confidence_idx]]
            confidence = scores[highest_confidence_idx]
            return detected_digit, confidence
        else:
            return None, 0

    def detect_letter(self, image):
        """
        Detect letter in an image using the preloaded letter recognition model.
        Returns the detected letter and its confidence.
        """
        input_image_width = config_charRec.input_w
        input_image_height = config_charRec.input_h

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_image_width, input_image_height))
        img_resized = img_resized.astype(np.uint8)
        img_input = np.expand_dims(img_resized, axis=0)

        outputs = self.rknn_letter.inference(inputs=[img_input])

        input_data = []
        LISTSIZE = 4 + 1 + config_charRec.NUM_CLS
        for output in outputs:
            batch, channels, height, width = output.shape
            SPAN = channels // LISTSIZE
            output = output.reshape(batch, SPAN, LISTSIZE, height, width)
            output = output.transpose(0, 3, 4, 1, 2)
            input_data.append(output[0])

        boxes, classes, scores = yolov3_post_process(
            input_data,
            config_charRec.anchors,
            config_charRec.masks,
            config_charRec.OBJ_THRESH,
            config_charRec.NMS_THRESH
        )

        if boxes is not None and len(boxes) > 0:
            highest_confidence_idx = np.argmax(scores)
            detected_letter = config_charRec.CLASSES[classes[highest_confidence_idx]]
            confidence = scores[highest_confidence_idx]
            return detected_letter, confidence
        else:
            return None, 0

    def license_plate_recognition_pipeline(self, image):
        """
        Complete pipeline for license plate recognition.
        Returns the recognized plate number and an overall confidence score.
        """
        if image is None:
            print('Error: Unable to read image file')
            return None, 0

        # Step 1: Detect license plate(s)
        plates, plate_scores = self.detect_plate(image)
        if not plates:
            print('No license plate detected')
            return None, 0

        # Use the best plate (highest score)
        best_plate = plates[0]
        plate_confidence = plate_scores[0]

        # Step 2: Segment characters from the best plate
        char_images = self.segment_chars(best_plate)
        if len(char_images) != 7:
            print(f'Expected 7 characters, but found {len(char_images)}')
            return None, 0

        # Step 3 & 4: Recognize characters
        plate_number = ''
        total_confidence = plate_confidence
        recognition_results = []

        # Recognize first 3 characters as letters
        for i in range(3):
            letter, confidence = self.detect_letter(char_images[i])
            if letter is not None:
                plate_number += letter
                total_confidence += confidence
                recognition_results.append((i, letter, confidence, 'letter'))
            else:
                plate_number += '?'

        # Recognize last 4 characters as digits
        for i in range(3, 7):
            digit, confidence = self.detect_digit(char_images[i])
            if digit is not None:
                plate_number += str(digit)
                total_confidence += confidence
                recognition_results.append((i, digit, confidence, 'digit'))
            else:
                plate_number += '?'

        # Calculate average confidence (including the plate detection)
        average_confidence = total_confidence / (len(recognition_results) + 1)
        return plate_number, average_confidence

    def release(self):
        """Release all RKNN resources when finished."""
        self.rknn_plate.release()
        self.rknn_char_seg.release()
        self.rknn_digit.release()
        self.rknn_letter.release()


if __name__ == '__main__':
    # Create an instance of the recognizer 
    recognizer = LicensePlateRecognizer()

    # For each image/frame you want to process:
    image = cv2.imread('path_to_your_image.jpg')
    plate_number, confidence = recognizer.license_plate_recognition_pipeline(image)
    print("Plate number:", plate_number)
    print("Confidence:", confidence)

    # When your application is done, release the models:
    recognizer.release()
