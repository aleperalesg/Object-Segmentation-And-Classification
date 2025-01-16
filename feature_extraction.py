import numpy as np
import os
import cv2

def get_paths(folder):
	paths = []

	for filename in os.listdir(folder):
			if filename.find('.DS') == -1:	## 
				paths.append(filename)
	
	return paths

def get_images(paths):
	imgs = []

	for i in paths:
		imgs.append(cv2.imread("segmentation/mask_dataset/"+i))

	return imgs

def find_equi_diameter(count): 
    area = cv2.contourArea(count) 
    equi_diameter = np.sqrt(4*area/np.pi) 
    return equi_diameter 




paths = get_paths("segmentation/mask_dataset/")
imgs = get_images(paths)

X = np.zeros((len(imgs),12))

for i in range(len(imgs)):

	## Feature vector of zeros
	aux = np.zeros((1,12))

	## Get binary images
	I = imgs[i]
	I = I[:,:,0]

	## Get contours
	contours, _ = cv2.findContours(I, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	## Get perimeter
	perimeter = cv2.arcLength(contours[0], True)
	## Get area
	area = cv2.contourArea(contours[0])
	# Get the equivalent diameter for this contour 
	elps =  cv2.fitEllipse(contours[0])
	diameter = max(elps[1]) 

	## Get shape features
	
	## Roundnes
	aux[0,0] = (4*area) / (np.pi*(diameter**2))
	## Compacity
	aux[0,1]  = 2 * np.sqrt(area * np.pi) / perimeter
	## Shape factor
	aux[0,2]=  4*area*np.pi / (perimeter**2)
	## 
	aux[0,3] = ((4*area) - perimeter) / 2


	## Get moments
	mu = cv2.moments(contours[0])

	## Get invariants moments
	xc = mu["m10"]/mu["m00"]
	yc = mu["m01"]/mu["m00"]
	den1 = mu["m00"]**2
	den2 = mu["m00"]**2.5

	## Normalized central moment 20
	m20 = mu["m20"] - xc * mu["m10"]
	m20c = m20/den1

	## Normalized central moment 02
	m02 = mu["m02"] - (yc*mu["m01"])
	m02c = m02/den1

	## Normalized central moment 11
	m11 = mu["m11"]- (xc * mu["m01"])
	m11c = m11/den1

	## Normalized central moment 21
	m21 = mu["m21"] - (2*xc*mu["m11"]) - (yc*mu["m20"]) + (2*mu["m01"] * xc**2 )
	m21c = m21/den2


	## Normalized central moment 12
	m12 = mu["m12"] - (2*yc* mu["m11"]) - (xc*mu["m02"]) + (2*mu["m10"] * yc**2 )
	m12c = m12/den2


	## Normalized central moment 30 
	m30 = mu["m30"]- 3*xc*mu["m20"]+ 2*mu["m10"]*xc**2
	m30c = m30/den2


	## Normalized central moment 03
	m03 =  mu["m03"]- 3*yc*mu["m02"] + 2*mu["m01"]*yc**2
	m03c = m03/den2


	## Hu moments 
	aux[0,4] = m02c + m20c
	aux[0,5] = (m20c -m02c)**2 + 4*m11c**2
	aux[0,6] = (m30c- 3*m12c)**2 + (3* m21c- m30c)
	aux[0,7] = (m30c + m12c)**2 + (m21c + m03c)**2
	aux[0,8] = (m30c - 3*m12c) * (m30c + m12c) * ((m30c + m12c)**2 - 3*(m21c + m03c)**2)  +  (3*m21c - m03c)*(m21c +m03c)* (3*(m30c + m12c)**2-( m21c + m03c)**2)
	aux[0,9] = (m20c - m02c)* ((m30c + m12c)**2-(m21c + m03c)**2) + 4*m11c*(m30c + m12c)*(m21c+m03c)
	aux[0,10] = (m21c - m03c) * (m30c + m12c)*((m30c + m12c)**2 - 3*(m21c+ m03c)**2) + (3*m12c - m30c) * (m21c + m03c) * (3*(m30c + m12c)**2 - (m21c + m03c)**2)


	## Get label class
	if not("control" not in paths[i]):
		aux[0,11] = 0 

	if not("orra" not in paths[i]):
		aux[0,11] = 1
		

	if not("raton" not in paths[i]):
		aux[0,11] = 2

	if not("reloj" not in paths[i]):
		aux[0,11] = 3

	X[i,:] = aux


np.save("features", X)









	


