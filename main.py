import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
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
input_shape = (224,224,3)
target_size = (224,224)

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
	train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=target_size,classes=['ad','mci','nc'],batch_size=10)
	valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=target_size,classes=['ad','mci','nc'],batch_size=10)
	test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=target_size,classes=['ad','mci','nc'],batch_size=10)

	#imgs,labels = next(train_batches)
	#plots(imgs,titles=labels)
	#model = Sequential([
			#Conv2D(,(3,3),activation='relu',input_shape=input_shape),
			#Conv2D(32,(3,3),activation='relu'),
			#MaxPooling2D(pool_size=(2,2)),
			#Dropout(0.25),
			#Flatten(),
			#Dense(3,activation='softmax'),
		#])


	vgg16 = keras.applications.vgg16.VGG16()
	model = Sequential()
	for layer in vgg16.layers:
		model.add(layer)

	model.layers.pop()
	for layer in model.layers:
		layer.trainable = False

	model.add(Dense(3,activation='softmax'))
	print("Resumo do modelo")
	print(model.summary())

	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
	model.fit_generator(train_batches,steps_per_epoch=150,validation_data=valid_batches,validation_steps=50,epochs=20,verbose=1)