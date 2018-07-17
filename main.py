import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras import optimizers
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

train_path = 'dataset/train'
valid_path = 'dataset/valid'
test_path = 'dataset/test'

def plots(ims,figsize=(12,6),rows=1,interp=False,titles=None):
	if type(ims[0]) is np.ndarray:
		ims = np.array(ims).astype(np.uint8)
		if (ims.shape[-1] != 3):
			ims = ims.transpose ((0,2,3,1))
	f = plt.figure(figsize=figsize)
	cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
	for i in range(len(ims)):
		sp = f.add_subplot(rows,cols,i+1)
		sp.axis('Off')
		if titles is not None:
			sp.set_title(titles[i],fontsize=16)
		plt.imshow(ims[i],interpolation=None if interp else 'none')
	plt.waitforbuttonpress()


if __name__ == "__main__":
	train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(256,166),classes=['ad','mci','nc'],batch_size=5)
	valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=(256,166),classes=['ad','mci','nc'],batch_size=2)
	test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(256,166),classes=['ad','mci','nc'],batch_size=2)

	#imgs,labels = next(train_batches)
	#plots(imgs,titles=labels)
	model = Sequential([
			Conv2D(32,(3,3),activation='relu',input_shape=(256,166,3)),
			Flatten(),
			Dense(3,activation='softmax'),
		])


	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])
	model.fit_generator(train_batches,steps_per_epoch=300,validation_data=valid_batches,validation_steps=250,epochs=2,verbose=2)