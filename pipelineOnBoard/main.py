import os
import numpy as np
import cv2
import time
from rknnlite.api import RKNNLite

from utils import yolov3_post_process, yolov3_post_process_char_seg
import config_charRec
import config_charSeg
import config_digitRec
import config_lpRec


def detect_plate(image, config):
    rknn = RKNNLite(verbose=True)
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

    detected_plates = []
    for box in boxes:
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

    return detected_plates, scores

def segment_chars(image,config):
    """
    Segment characters from the input image.
    Returns up to 7 highest-scoring character detections.
    """
    # Initialize RKNN
    rknn = RKNNLite(verbose=True)
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

    # Crop characters
    cropped_chars = []
    for box in boxes:
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
    ret = rknn.load_rknn(config.MODEL_PATH)
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

import os
import numpy as np
import cv2
import time
from rknnlite.api import RKNNLite

from utils import yolov3_post_process, yolov3_post_process_char_seg
import config_charRec
import config_charSeg
import config_digitRec
import config_lpRec


def detect_plate(image, config):
    rknn = RKNNLite(verbose=True)
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

    detected_plates = []
    for box in boxes:
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

    return detected_plates, scores

def segment_chars(image,config):
    """
    Segment characters from the input image.
    Returns up to 7 highest-scoring character detections.
    """
    # Initialize RKNN
    rknn = RKNNLite(verbose=True)
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

    # Crop characters
    cropped_chars = []
    for box in boxes:
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
    ret = rknn.load_rknn(config.MODEL_PATH)
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

def license_plate_recognition_pipeline(image_path, debug=False):
    """
    Complete pipeline for license plate recognition with debug image saving.
    
    Args:
        image_path: Path to the input image
        debug: Boolean to enable/disable debug image saving
    
    Returns:
        plate_number: Recognized license plate number as string
        confidence: Overall confidence score
    """
    # Create debug directories if debug mode is enabled
    if debug:
        debug_dirs = create_debug_directories()
        
    # Get filename without extension for debug saving
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print('Error: Unable to read image file')
        return None, 0

    if debug:
        cv2.imwrite(os.path.join(debug_dirs['main'], f'{base_filename}_original.jpg'), image)

    # Step 1: Detect license plate
    plates, plate_scores = detect_plate(image, config_lpRec)
    if not plates:
        print('No license plate detected')
        return None, 0

    # Save detected plates
    if debug:
        for idx, (plate, score) in enumerate(zip(plates, plate_scores)):
            plate_path = os.path.join(debug_dirs['plates'], 
                                    f'{base_filename}_plate_{idx}_score_{score:.2f}.jpg')
            cv2.imwrite(plate_path, plate)

    # Use the plate with highest confidence
    best_plate = plates[0]
    plate_confidence = plate_scores[0]

    # Step 2: Segment characters
    char_images = segment_chars(best_plate, config_charSeg)
    if len(char_images) != 7:  # Expecting 7 characters
        print(f'Expected 7 characters, but found {len(char_images)}')
        
    # Save segmented characters
    if debug:
        for idx, char_img in enumerate(char_images):
            char_path = os.path.join(debug_dirs['chars'], 
                                   f'{base_filename}_char_{idx}.jpg')
            cv2.imwrite(char_path, char_img)

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
                
                if debug:
                    letter_path = os.path.join(debug_dirs['letters'], 
                                             f'{base_filename}_letter_{i}_{letter}_{confidence:.2f}.jpg')
                    cv2.imwrite(letter_path, char_images[i])
            else:
                plate_number += '?'

    # Process last 4 characters as digits
    for i in range(3, 7):
        if i < len(char_images):
            digit, confidence = detect_digit(char_images[i], config_digitRec)
            if digit is not None:
                plate_number += str(digit)
                total_confidence += confidence
                recognition_results.append((i, digit, confidence, 'digit'))
                
                if debug:
                    digit_path = os.path.join(debug_dirs['digits'], 
                                            f'{base_filename}_digit_{i}_{digit}_{confidence:.2f}.jpg')
                    cv2.imwrite(digit_path, char_images[i])
            else:
                plate_number += '?'

    # Calculate average confidence
    average_confidence = total_confidence / (len(recognition_results) + 1)  # +1 for plate detection

    # Save recognition summary
    if debug:
        summary_path = os.path.join(debug_dirs['main'], f'{base_filename}_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"License Plate Recognition Summary\n")
            f.write(f"================================\n")
            f.write(f"Input Image: {image_path}\n")
            f.write(f"Detected Plate Number: {plate_number}\n")
            f.write(f"Average Confidence: {average_confidence:.2f}\n\n")
            f.write(f"Plate Detection Score: {plate_confidence:.2f}\n\n")
            f.write("Character Recognition Details:\n")
            for pos, char, conf, char_type in recognition_results:
                f.write(f"Position {pos}: {char} ({char_type}) - Confidence: {conf:.2f}\n")

    return plate_number, average_confidence




# Example usage
if __name__ == '__main__':
    image_path = './car3.png'
    
    start_time = time.time()
    plate_number, confidence = license_plate_recognition_pipeline(image_path, debug = True)
    processing_time = time.time() - start_time

    if plate_number:
        print(f'Detected License Plate: {plate_number}')
        print(f'Confidence: {confidence:.2f}')
        print(f'Processing Time: {processing_time:.2f} seconds')
    else:
        print('Failed to detect license plate')def license_plate_recognition_pipeline(image_path, debug=False):
    """
    Complete pipeline for license plate recognition with debug image saving.
    
    Args:
        image_path: Path to the input image
        debug: Boolean to enable/disable debug image saving
    
    Returns:
        plate_number: Recognized license plate number as string
        confidence: Overall confidence score
    """
    # Create debug directories if debug mode is enabled
    if debug:
        debug_dirs = create_debug_directories()
        
    # Get filename without extension for debug saving
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print('Error: Unable to read image file')
        return None, 0

    if debug:
        cv2.imwrite(os.path.join(debug_dirs['main'], f'{base_filename}_original.jpg'), image)

    # Step 1: Detect license plate
    plates, plate_scores = detect_plate(image, config_lpRec)
    if not plates:
        print('No license plate detected')
        return None, 0

    # Save detected plates
    if debug:
        for idx, (plate, score) in enumerate(zip(plates, plate_scores)):
            plate_path = os.path.join(debug_dirs['plates'], 
                                    f'{base_filename}_plate_{idx}_score_{score:.2f}.jpg')
            cv2.imwrite(plate_path, plate)

    # Use the plate with highest confidence
    best_plate = plates[0]
    plate_confidence = plate_scores[0]

    # Step 2: Segment characters
    char_images = segment_chars(best_plate, config_charSeg)
    if len(char_images) != 7:  # Expecting 7 characters
        print(f'Expected 7 characters, but found {len(char_images)}')
        
    # Save segmented characters
    if debug:
        for idx, char_img in enumerate(char_images):
            char_path = os.path.join(debug_dirs['chars'], 
                                   f'{base_filename}_char_{idx}.jpg')
            cv2.imwrite(char_path, char_img)

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
                
                if debug:
                    letter_path = os.path.join(debug_dirs['letters'], 
                                             f'{base_filename}_letter_{i}_{letter}_{confidence:.2f}.jpg')
                    cv2.imwrite(letter_path, char_images[i])
            else:
                plate_number += '?'

    # Process last 4 characters as digits
    for i in range(3, 7):
        if i < len(char_images):
            digit, confidence = detect_digit(char_images[i], config_digitRec)
            if digit is not None:
                plate_number += str(digit)
                total_confidence += confidence
                recognition_results.append((i, digit, confidence, 'digit'))
                
                if debug:
                    digit_path = os.path.join(debug_dirs['digits'], 
                                            f'{base_filename}_digit_{i}_{digit}_{confidence:.2f}.jpg')
                    cv2.imwrite(digit_path, char_images[i])
            else:
                plate_number += '?'

    # Calculate average confidence
    average_confidence = total_confidence / (len(recognition_results) + 1)  # +1 for plate detection

    # Save recognition summary
    if debug:
        summary_path = os.path.join(debug_dirs['main'], f'{base_filename}_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"License Plate Recognition Summary\n")
            f.write(f"================================\n")
            f.write(f"Input Image: {image_path}\n")
            f.write(f"Detected Plate Number: {plate_number}\n")
            f.write(f"Average Confidence: {average_confidence:.2f}\n\n")
            f.write(f"Plate Detection Score: {plate_confidence:.2f}\n\n")
            f.write("Character Recognition Details:\n")
            for pos, char, conf, char_type in recognition_results:
                f.write(f"Position {pos}: {char} ({char_type}) - Confidence: {conf:.2f}\n")

    return plate_number, average_confidence




# Example usage
if __name__ == '__main__':
    image_path = './car3.png'
    
    start_time = time.time()
    plate_number, confidence = license_plate_recognition_pipeline(image_path, debug = True)
    processing_time = time.time() - start_time

    if plate_number:
        print(f'Detected License Plate: {plate_number}')
        print(f'Confidence: {confidence:.2f}')
        print(f'Processing Time: {processing_time:.2f} seconds')
    else:
        print('Failed to detect license plate')