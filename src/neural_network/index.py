import os
import numpy as np
import cv2
import time
from rknnlite.api import RKNNLite

from neural_network.utils import yolov3_post_process, yolov3_post_process_char_seg
import neural_network.config_charRec as config_charRec
import neural_network.config_charSeg as config_charSeg
import neural_network.config_digitRec as config_digitRec
import neural_network.config_lpRec as config_lpRec


def detect_plate(image, config):
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(config.RKNN_MODEL_PATH) != 0:
        print('Load RKNN model failed!')
        return [], []
    
    if rknn.init_runtime() != 0:
        print('Init runtime environment failed!')
        return [], []

    img = cv2.resize(image, (416, 448))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb, axis=0)
    img = img.astype(np.uint8)

    outputs = rknn.inference(inputs=[img_input])
    rknn.release()

    input_data = []
    for output in outputs:
        batch, channels, height, width = output.shape
        SPAN = channels // config.LISTSIZE
        output = output.reshape(batch, SPAN, config.LISTSIZE, height, width)
        output = output.transpose(0, 3, 4, 1, 2)
        output = output[0]
        input_data.append(output)

    boxes, classes, scores = yolov3_post_process(input_data, config.anchors, config.masks, config.OBJ_THRESH, config.NMS_THRESH)
    if boxes is None:
        return [], []
    
    # Create a list of detections (box, score)
    detections = list(zip(boxes, scores))

    # Order top detections
    top_detections = sorted(detections, key=lambda x: float(x[1]), reverse=True)

    detected_plates = []
    sorted_scores = []
    for box, score in top_detections:
        x, y, w, h = box
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]

        # Expand box by 20% (10% on each side)
        expand_x = int(w * 0.1)  # 10% of width
        expand_y = int(h * 0.1)  # 10% of height

        left = max(0, int(x - expand_x))
        top = max(0, int(y - expand_y))
        right = min(image.shape[1], int(x + w + expand_x))
        bottom = min(image.shape[0], int(y + h + expand_y))

        detected_plates.append(image[top:bottom, left:right])
        sorted_scores.append(score)

    return detected_plates, sorted_scores

def segment_chars(image,config):
    """
    Segment characters from the input image.
    Returns up to 7 highest-scoring character detections.
    """
    # Initialize RKNN
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(config.RKNN_MODEL_PATH)
    if ret != 0:
        print('Load RKNN model failed!')
        return []
    
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        return []

    # Preprocess image
    img = cv2.resize(image, (256, 96))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb, axis=0)

    # Inference
    outputs = rknn.inference(inputs=[img_input])

    # Process outputs
    input_data = []
    for output in outputs:
        batch, channels, height, width = output.shape
        SPAN = channels // config.LISTSIZE
        output = output.reshape(batch, SPAN, config.LISTSIZE, height, width)
        output = output.transpose(0, 3, 4, 1, 2)
        output = output[0]
        input_data.append(output)

    # Use the char-specific post-processing
    boxes, classes, scores = yolov3_post_process_char_seg(input_data, config.anchors, config.masks, config.OBJ_THRESH, config.NMS_THRESH)
    
    rknn.release()

    if boxes is None:
        print("No characters detected")
        return []

    # Create a list of detections (box, score)
    detections = list(zip(boxes, scores))

    # Extract top 7 highest-scoring detections
    top_detections = sorted(detections, key=lambda x: float(x[1]), reverse=True)[:7]

    # Sort detections left to right (by x-coordinate)
    top_detections = sorted(top_detections, key=lambda x: float(x[0][0]))

    # Crop and process selected characters
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

def detect_digit(img, config):
    """
    Detect digit in an image using RKNN model.
    
    Args:
        img: Numpy array of the image
        config: Configuration module containing model path and other constants
    
    Returns:
        detected_digit: The detected digit (0-9) or None if no digit is detected
        confidence: Confidence score of the detection
    """
    # Create RKNN object
    rknn = RKNNLite(verbose=False)

    # Load RKNN model
    ret = rknn.load_rknn(config.RKNN_MODEL_PATH)
    if ret != 0:
        print('Load RKNN model failed!')
        return None, 0

    # Set input dimensions from config
    input_image_width = config.input_w
    input_image_height = config.input_h

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_image_width, input_image_height))
    img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)

    # Init runtime environment
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        return None, 0

    # Inference
    outputs = rknn.inference(inputs=[img])

    # Process outputs
    input_data = []
    LISTSIZE = 4 + 1 + config.NUM_CLS
    for output in outputs:
        batch, channels, height, width = output.shape
        SPAN = channels // LISTSIZE
        output = output.reshape(batch, SPAN, LISTSIZE, height, width)
        output = output.transpose(0, 3, 4, 1, 2)
        output = output[0]
        input_data.append(output)

    # Post-processing
    boxes, classes, scores = yolov3_post_process(input_data, config.anchors, config.masks, config.OBJ_THRESH, config.NMS_THRESH)

    # Release RKNN runtime
    rknn.release()

    # Return results
    if boxes is not None and len(boxes) > 0:
        # Return the digit with highest confidence
        highest_confidence_idx = np.argmax(scores)
        detected_digit = config.CLASSES[classes[highest_confidence_idx]]
        confidence = scores[highest_confidence_idx]
        return detected_digit, confidence
    else:
        return None, 0

def detect_letter(image, config):
    """
    Detect letter in an OpenCV image using RKNN model.
    
    Args:
        image: OpenCV image (numpy array in BGR format)
        model_path: Path to the RKNN model file
    
    Returns:
        detected_letter: The detected letter (A-Z) or None if no letter is detected
        confidence: Confidence score of the detection
    """
    # Create RKNN object
    rknn = RKNNLite(verbose=False)

    # Load RKNN model
    ret = rknn.load_rknn(config.RKNN_MODEL_PATH)
    if ret != 0:
        print('Load RKNN model failed!')
        return None, 0

    # Set input dimensions
    input_image_width = config.input_w
    input_image_height = config.input_h

    # Preprocess the image
    if image is None:
        print('Error: Invalid image input')
        return None, 0

    # Convert BGR to RGB and resize
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_image_width, input_image_height))
    img = img.astype(np.uint8)
    img = np.expand_dims(img, axis=0)

    # Init runtime environment
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        return None, 0

    # Inference
    outputs = rknn.inference(inputs=[img])

    # Process outputs
    input_data = []
    LISTSIZE = 4 + 1 + config.NUM_CLS
    for output in outputs:
        batch, channels, height, width = output.shape
        SPAN = channels // LISTSIZE
        output = output.reshape(batch, SPAN, LISTSIZE, height, width)
        output = output.transpose(0, 3, 4, 1, 2)
        output = output[0]
        input_data.append(output)

    # Post-processing
    boxes, classes, scores = yolov3_post_process(input_data, config.anchors, config.masks, config.OBJ_THRESH, config.NMS_THRESH)

    # Release RKNN runtime
    rknn.release()

    # Return results
    if boxes is not None and len(boxes) > 0:
        # Return the letter with highest confidence
        highest_confidence_idx = np.argmax(scores)
        detected_letter = config.CLASSES[classes[highest_confidence_idx]]
        confidence = scores[highest_confidence_idx]
        return detected_letter, confidence
    else:
        return None, 0

def create_debug_directories():
    """Create directories for debug images"""
    debug_dirs = {
        'main': 'debug_output',
        'plates': 'debug_output/1_detected_plates',
        'chars': 'debug_output/2_segmented_chars',
        'letters': 'debug_output/3_recognized_letters',
        'digits': 'debug_output/4_recognized_digits'
    }
    
    for dir_path in debug_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return debug_dirs

def license_plate_recognition_pipeline(image):
    """
    Complete pipeline for license plate recognition with debug image saving.
    
    Args:
        image: The frame of the video
        debug: Boolean to enable/disable debug image saving
    
    Returns:
        plate_number: Recognized license plate number as string
        confidence: Overall confidence score
    """
    
    # Read the image
    if image is None:
        print('Error: Unable to read image file')
        return None, 0

    # Step 1: Detect license plate
    plates, plate_scores = detect_plate(image, config_lpRec)
    if not plates:
        print('No license plate detected')
        return None, 0

    print('detectou placa')

    # Use the plate with highest confidence
    best_plate = plates[0]
    plate_confidence = plate_scores[0]

    # Step 2: Segment characters
    char_images = segment_chars(best_plate, config_charSeg)
    if len(char_images) != 7:  # Expecting 7 characters
        print(f'Expected 7 characters, but found {len(char_images)}')
        return None, 0

    print('segmentou placa')

    # Step 3 & 4: Recognize characters
    plate_number = ''
    total_confidence = plate_confidence
    recognition_results = []

    # Process first 3 characters as letters
    for i in range(3):
        if i < len(char_images):
            letter, confidence = detect_letter(char_images[i], config_charRec)
            if letter is not None:
                plate_number += letter
                total_confidence += confidence
                recognition_results.append((i, letter, confidence, 'letter'))
            else:
                plate_number += '?'
    print('reconheceu letras')

    # Process last 4 characters as digits
    for i in range(3, 7):
        if i < len(char_images):
            digit, confidence = detect_digit(char_images[i], config_digitRec)
            if digit is not None:
                plate_number += str(digit)
                total_confidence += confidence
                recognition_results.append((i, digit, confidence, 'digit'))
            else:
                plate_number += '?'
    print('reconheceu digitos')

    # Calculate average confidence
    average_confidence = total_confidence / (len(recognition_results) + 1)  # +1 for plate detection

    return plate_number, average_confidence