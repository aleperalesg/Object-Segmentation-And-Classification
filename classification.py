import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Activation 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
## z-score normalization
def zscore(X,stats = False):

	## sentence for training set
	if isinstance(stats, bool):
		mn = np.mean(X,axis = 0)
		std = np.std(X,axis = 0)
		stats = np.array([mn, std])
		X = (X[...,:] - stats[0,:])/stats[1,:]
		return X,stats

	## sentence for test set
	else:
		X = (X[...,:] - stats[0,:])/stats[1,:]
		return X


## load dataset
X = np.load("features.npy")


#X = np.delete(X, 3, 1) 
Y = X[:,-1]
X = X = np.delete(X, 2, 1) 
X = X[:,0:3]


Y = Y[:,np.newaxis]



## k-fold cross validation 
k = 5 
skf = StratifiedKFold(n_splits=k,shuffle = False)
skf.get_n_splits(X, Y)
cm = []
acc = np.zeros((5,2))
sum_cm = np.zeros((4,4))
for i, (train_index, test_index) in enumerate(skf.split(X,Y)):
	

	Xtr = X[train_index]
	Ytr = Y[train_index]

	Xtt = X[test_index]
	Ytt = Y[test_index]

	Xtr,stats = zscore(Xtr)
	Xtt = zscore(Xtt,stats)

	model = Sequential([ 
	    
	    # reshape 28 row * 28 column data to 28*28 rows 
	    Dense(10,activation = 'relu'), 
	    
	      # dense layer 1 
	    Dense(256, activation='relu'),   
	    
	    # dense layer 2 
	    Dense(128, activation='relu'),  
	    
	      # output layer 
	    Dense(4, activation='softmax'),   
	]) 


	model.compile(optimizer='adam', 
	              loss='sparse_categorical_crossentropy', 
	              metrics=['accuracy']) 

	model.fit(Xtr, Ytr, epochs=20,  
	          batch_size=1,  
	          validation_split=0.2) 



	Ypp = model.predict(Xtt)
	Ypp = np.argmax(Ypp,axis = 1)





	acc[i,:] = model.evaluate(Xtt,  Ytt, verbose = 0)
	cma = confusion_matrix(Ytt, Ypp)
	sum_cm +=cma
	cm.append(cma)



###################################################################################

confusion_matrix = sum_cm/k
suma_cm = confusion_matrix.sum(axis = 1, keepdims =True)
confusion_matrix = confusion_matrix/suma_cm




print('test loss, test acc:', acc)
print(np.mean(acc,0))


## Display confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Controller','Cap','Mouse','Watch'])
cm_display.plot()
plt.show()







