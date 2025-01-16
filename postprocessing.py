import numpy as np
import cv2
import os


def get_paths(folder):
	paths = []
	for filename in os.listdir(folder):
		if filename.find('.DS') == -1:	## 
			paths.append(filename)
	return paths

def get_images(paths):
	images = []
	for i in paths:
		img  = cv2.imread("segmentation//prev_mask_dataset//"+i)
		img = img[:,:,1] 
		images.append(img)
	return images


## Get images paths
paths = get_paths("segmentation//prev_mask_dataset//")
mask_dataset = get_images(paths)

## Get structuran element for morphological operaors
str_ele = morph_rect = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(7,7))

## Postprocessing loop
for k in range(len(mask_dataset)):



    ## Get images from batch
    I = mask_dataset[k] 
    img = np.zeros((I.shape[0],I.shape[1],3))

    ## Get contours from binary images
    cs,w = cv2.findContours(I, mode=cv2.RETR_EXTERNAL ,method=cv2.CHAIN_APPROX_SIMPLE)

    ## Get largest contour
    aux = np.zeros(len(cs))
    for i in range(len(cs)):
    	aux[i] = cv2.contourArea(cs[i])

    ## Get object from binary image
    con = cv2.drawContours(image=img, contours=[cs[np.argmax(aux)]], contourIdx=-1, color=(255, 255, 255),thickness=-1)
    con = con[:,:,0]

    ## Opening operator that consist of applie dilation followed by erosion
    con = cv2.erode(con, kernel=str_ele, iterations=1)
    con = cv2.dilate(con, kernel=str_ele, iterations=1)


    ## Fill holes
    con = con.astype(np.uint8)
    cs2,_ = cv2.findContours(con, mode=cv2.RETR_EXTERNAL ,method=cv2.CHAIN_APPROX_SIMPLE)
    img2 = np.zeros((I.shape[0],I.shape[1],3))
    largest_contour = max(cs2, key=cv2.contourArea)
    con = cv2.drawContours(image=img2, contours=[largest_contour], contourIdx=-1, color=(255, 255, 255),thickness=-1)
    

    ## Save binary mask 
    cv2.imwrite("segmentation/mask_dataset/"+paths[k],con)




