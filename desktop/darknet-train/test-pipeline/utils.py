import numpy as np

def dknet_label_reverse(R,img_width,img_height):
	WH = np.array([img_width,img_height],dtype=float)
	L  = []
	for r in R:
		center = np.array(r[2][:2])/WH
		wh2 = (np.array(r[2][2:])/WH)*.5
		L.append(Label(ord(r[0]),tl=center-wh2,br=center+wh2,prob=r[1]))
	return L
