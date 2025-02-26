import torch
# torch.device('cpu')
torch.cuda.set_device(0)  # Set to your desired GPU number
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device='cpu'
print(f'Using device: {device}')
print(torch.cuda.get_device_name())

import sys, os
from tensorflow import keras
import cv2
import traceback
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json

# import tensorflow as tf
# tf.test.is_built_with_cuda()
# tf.config.list_physical_devices('GPU')




from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes
from time import sleep
import datetime
import numpy as np
def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

from ultralytics import YOLO



if __name__ == '__main__':

	try:
		
		input_dir  = sys.argv[1]
		output_dir = input_dir
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)		
		lp_threshold = .5

		wpod_net_path = sys.argv[2]
		print(wpod_net_path)
		# wpod_net = load_model(path=wpod_net_path,custom_objects={'BatchNormalization.': keras.layers.BatchNormalization})

		imgs_paths = glob('%s/*.png' % input_dir)

		print ('Searching for license plates using WPOD-NET')
		alpr_path='../Automatic-License-Plate-Recognition-using-YOLOv8/'
		license_plate_detector = YOLO(alpr_path+'license_plate_detector.pt')  # License plate detection model


		for i,img_path in enumerate(imgs_paths):

			print ('\t Processing %s' % img_path)
			start = datetime.datetime.now()
			bname = splitext(basename(img_path))[0]
			Ivehicle = cv2.imread(img_path)

			ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
			side  = int(ratio*288.)
			bound_dim = min(side + (side%(2**4)),608)
			print( "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

			# Llp,LlpImgs,_ = detect_lp(license_plate_detector,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold,device)
			license_plates = license_plate_detector.predict(source=Ivehicle,device=device,verbose=False)[0]
			i=0
			for license_plate in license_plates.boxes.data.tolist():
				x1, y1, x2, y2, score, class_id = license_plate
				license_plate_crop = Ivehicle[int(y1):int(y2), int(x1): int(x2), :]
				license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
				# license_plate_crop_gray= cv2.cvtColor(license_plate_crop_gray, cv2.COLOR_GRAY2BGR)
				# writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
				stop = datetime.datetime.now()
				print(stop-start)
				img=license_plate_crop
				img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				kernel = np.ones((1, 1, 1), np.uint8)
				img = cv2.dilate(img, kernel, iterations=1)
				img = cv2.erode(img, kernel, iterations=1)
				# im2=cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
				im5=cv2.adaptiveThreshold(cv2.bilateralFilter(img, 5, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
				cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),img)
				s = Shape()
				writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])




			# if len(license_plates):
			# 	Ilp = license_plates[0]
			# 	Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
			# 	Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

			# 	s = Shape(license_plates[0].pts)

			# 	cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
				writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
			stop = datetime.datetime.now()
			print(stop-start)

	except:
		traceback.print_exc()
		sys.exit(1)
	sleep(10)
	sys.exit(0)


