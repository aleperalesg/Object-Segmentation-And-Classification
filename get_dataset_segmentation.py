import numpy as np
import cv2
import os
###################################################################
"""
 The dataset is composed by images with white background therefore we can  segment the 
 dataset by semantic segmentation, this means that we gonna classify each pixel of the 
 images to set as an object or background.


 By converting rgb images to ycbcr to images we can classify each pixel according to 
 is components or channels Y Cb Cr
"""
#####################################################################


def get_paths(folder):
	paths = []
	for filename in os.listdir(folder):
		if filename.find('.DS') == -1:	## 
			paths.append(filename)
	return paths

def get_images(paths):
	images = []
	for i in paths:
		img  = cv2.imread("dataset//"+i)
		sizex  = int(img.shape[0]*0.25)
		sizey  = int(img.shape[1]*0.25)
		img = cv2.resize(img,(sizey,sizex))
		images.append(img)

	return images

def bgr2ycbcr(images):
	ycbcr_imgs = []
	for i in range(len(images)):
		img = cv2.cvtColor(images[i], cv2.COLOR_BGR2YCrCb)
		#img = cv2.GaussianBlur(img,(5,5),1)
		ycbcr_imgs.append(img)
	

	return ycbcr_imgs	


def semantic_segmentation(ycbcr_imgs,paths,mn,inv,det):
	## bayesian classifier for semantic segmentation
	## mn -> mean, inv -> inverse of matrix covariance , det -> determinant of covaraince matrix
	## loop for list of images
	for k in range(len(ycbcr_imgs)):
		## get image
		prueba = ycbcr_imgs[k]
		## create binary image
		M,N= prueba.shape[0], prueba.shape[1]
		seg = np.zeros((M,N))



		cons = 1 / (np.sqrt((2 * np.pi) ** np.mean(mn) * det))

		#prueba  =  cv2.GaussianBlur(prueba, (3, 3), sigmaX=1,sigmaY=1)

		checkkkkkk = []
		p = []
		for i in range(M):
			for j in range(N):
				## get pixels i,j from each channel 
				
				
				pixels = np.array([ prueba[i,j,0], prueba[i,j,1], prueba[i,j,2]])
				pixels = pixels[:,np.newaxis]

				## evaluate pixels on a normal distribution with parameters: mn, inv, det
				aux = np.exp(-0.5 * (pixels-mn).T @ inv @ (pixels - mn))

				p.append(aux)
				## discriminant function  
				if aux < 0.145:
					seg[i,j] = 1
				else: 
					seg[i,j] = 0


		## save mask get_images
		path =  paths[k][:-4]
		cv2.imwrite("segmentation//prev_mask_dataset//"+path+".png",seg*255)





## get images paths
paths = get_paths("dataset")

## read and convert images 
imgs = get_images(paths)
ycbcr_imgs = bgr2ycbcr(imgs)




## I choose 20 images to crop the background and use to train a bayesin classifier 
X = np.zeros((1,3))
flag = False
cropped_imagep = []
for i in range(len(paths)):

	print(paths[i])

	if paths[i] == "reloj_001.jpg":
		cropped_image = ycbcr_imgs[i][0:253,300:362]
		flag = True


	if paths[i] == "reloj_065.jpg":

		cropped_image = ycbcr_imgs[i][0:253,0:190]
		flag = True

	if paths[i] == "reloj_033.jpg":
		cropped_image = ycbcr_imgs[i][0:253,0:120]
		flag = True

	if paths[i] == "reloj_032.jpg":
		cropped_image = ycbcr_imgs[i][80:253,0:200]

		cropped_imagep.append(ycbcr_imgs[i][80:253,0:200])
		cropped_imagep.append(ycbcr_imgs[i][80:253,0:200])		
		flag = True


	if paths[i] == "reloj_031.jpg":
		cropped_image = ycbcr_imgs[i][0:130,0:160]

		cropped_imagep.append(ycbcr_imgs[i][0:130,0:160])
		cropped_imagep.append(ycbcr_imgs[i][0:130,0:160])


		flag = True

	if paths[i] == "reloj_013.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:180]
		flag = True

	if paths[i] == "reloj_012.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:180]
		flag = True

	if paths[i] == "reloj_009.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:120]
		flag = True

	if paths[i] == "reloj_012.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:140]
		cropped_imagep.append(ycbcr_imgs[i][0:253, 0:140])
		flag = True

	if paths[i] == "reloj_042.jpg":
		cropped_image = ycbcr_imgs[i][100:253, 0:363]
		flag = True

	if paths[i] == "reloj_016.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:135]
		flag = True

	if paths[i] == "reloj_015.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:170]
		flag = True

	if paths[i] == "reloj_017.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:140]
		flag = True


	if paths[i] == "reloj_039.jpg":

		cropped_image = ycbcr_imgs[i][90:253, 180:365]


		flag = True


#################################################################################


	if paths[i] == "control_015.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:150]
		flag = True
	
	if paths[i] == "control_018.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:160]
		flag = True



	if paths[i] == "control_004.jpg":
		cropped_image = ycbcr_imgs[i][100:253, 0:363]
		flag = True


#########################################################################################

	if paths[i] == "gorra_054.jpg":
		
		cropped_image = ycbcr_imgs[i][0:253, 0:750]
		cropped_imagep.append(ycbcr_imgs[i][0:253, 0:750])
		flag = True



	if paths[i] == "gorra_041.jpg":
		cropped_image = ycbcr_imgs[i][200:253, 0:70]
		flag = True


	if paths[i] == "gorra_056.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:110]

		cropped_imagep.append(ycbcr_imgs[i][0:253, 0:110])
		cropped_imagep.append(ycbcr_imgs[i][0:253, 0:110])
		cropped_imagep.append(ycbcr_imgs[i][0:253, 0:110])
		cropped_imagep.append(ycbcr_imgs[i][0:253, 0:110])
		cropped_imagep.append(ycbcr_imgs[i][0:253, 0:110])

		flag = True

	if paths[i] == "gorra_046.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:125]
		flag = True

	if paths[i] == "gorra_051.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 185:363]
		cropped_imagep.append(ycbcr_imgs[i][0:253, 185:363])
		flag = True



	if paths[i] == "gorra_043.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 160:363]
		flag = True



	if paths[i] == "gorra_048.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:160]	
		
		cropped_imagep.append(ycbcr_imgs[i][120:253, 0:230])
		cropped_imagep.append(ycbcr_imgs[i][120:253, 0:230])
		flag = True

	if paths[i] == "gorra_049.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:150]
		flag = True

##################################################################################

	if paths[i] == "raton_036.jpg" :
		cropped_image = ycbcr_imgs[i][0:253, 0:240]
		flag = True
	

	if paths[i] == "raton_044.jpg" :
		cropped_image = ycbcr_imgs[i][20:90, 130:363]
		flag = True

	
	if paths[i] == "raton_002.jpg":
		cropped_image = ycbcr_imgs[i][0:253,200:362]
		flag = True

	if paths[i] == "raton_014.jpg":
		cropped_image = ycbcr_imgs[i][80:253, 0:362]
		flag = True



	if paths[i] == "raton_064.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:200]
		flag = True


	if paths[i] == "raton_063.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:190]
		cropped_imagep.append(ycbcr_imgs[i][0:253, 0:190])
		flag = True


	if paths[i] == "raton_028.jpg":
		cropped_image = ycbcr_imgs[i][0:253, 0:180]

		cropped_imagep.append(ycbcr_imgs[i][0:253, 0:180])
		flag = True


	if paths[i] == "raton_045.jpg":
		
		cropped_image = ycbcr_imgs[i][140:253, 190:362]
		cropped_imagep.append(ycbcr_imgs[i][140:253, 190:362])

		flag = True


	if paths[i] == "raton_050.jpg":

		cropped_image = ycbcr_imgs[i][75:253, 190:362]
		flag = True


	if paths[i] == "raton_057.jpg":
		cropped_image = ycbcr_imgs[i][20:125, 190:362]
		flag = True


	if paths[i] == "raton_055.jpg":

		cropped_image = ycbcr_imgs[i][75:120, 190:362]
		flag = True




	if paths[i] == "raton_054.jpg":
		cropped_image = ycbcr_imgs[i][126:180, 200:362]
		flag = True


	if paths[i] == "raton_041.jpg":
		cropped_image = ycbcr_imgs[i][70:95, 200:362]
		flag = True



	if paths[i] == "raton_065.jpg":
		cropped_image = ycbcr_imgs[i][100:120, 180:362]
		flag = True



	if paths[i] == "raton_069.jpg":
		cropped_image = ycbcr_imgs[i][45:115, 240:362]
		flag = True




	if paths[i] == "raton_070.jpg":
		cropped_image = ycbcr_imgs[i][80:130, 240:362]
		flag = True


	if paths[i] == "raton_056.jpg":
		cropped_image = ycbcr_imgs[i][65:95, 250:362]
		flag = True

#############################################################
	


	if flag == True:
		## get channels of ycbcr image
		x1 = cropped_image[:,:,0]
		x2 = cropped_image[:,:,1]
		x3 = cropped_image[:,:,2]

		## flatten channels
		x1 = x1.flatten()
		x2 = x2.flatten()
		x3 = x3.flatten()

		## create X array to train bayesian classifier
		Xaux  = np.column_stack((x1,x2,x3))
		X = np.concatenate([X,Xaux], axis=0)
		flag = False

## add cropped_image2 to X array
X = (np.delete(X, 0, 0))

for i in range(len(cropped_imagep)):
	print(i)
	cropped_image = cropped_imagep[i]
	x1 = cropped_image[:,:,0]
	x2 = cropped_image[:,:,1]
	x3 = cropped_image[:,:,2]
	x1 = x1.flatten()
	x2 = x2.flatten()
	x3 = x3.flatten()
	Xaux  = np.column_stack((x1,x2,x3))
	X = np.concatenate([X,Xaux], axis=0)
	X = np.concatenate([X,Xaux], axis=0)
	X = np.concatenate([X,Xaux], axis=0)
	X = np.concatenate([X,Xaux], axis=0)
	X = np.concatenate([X,Xaux], axis=0)
	X = np.concatenate([X,Xaux], axis=0)
	X = np.concatenate([X,Xaux], axis=0)



#X = X[:,[1,2]]


## mean of X by channel
mn = np.mean(X,axis =0)
mn = mn[:,np.newaxis]	

## covariance of X
cov = np.cov(X.T)
## inverse of covariance matrix of X
inv = np.linalg.inv(cov)
## determiant of covariance matrix of X
dete = np.linalg.det(cov)


semantic_segmentation(ycbcr_imgs,paths,mn,inv,dete)









		


