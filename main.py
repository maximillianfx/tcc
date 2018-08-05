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
from keras import regularizers
from keras.layers.convolutional import *
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import string
import ipdb

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0)

config = tf.ConfigProto(intra_op_parallelism_threads=24,inter_op_parallelism_threads=2, allow_soft_placement=True,device_count = {'CPU': 24 })
session = tf.Session(config=config)
K.set_session(session)
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

train_path = 'dataset-cropped-nedge/train'
valid_path = 'dataset-cropped-nedge/valid'
test_path = 'dataset-cropped-nedge/test'
DATA_SET_EXECUTE = ['ORIGINAL','CROPPED_WITHOUT_EDGE','CROPPED_EDGE']
NUMERO_EPOCHS = 50
input_shape = (224,224,3)
target_size = (224,224)


if __name__ == "__main__":
	os.system('cls' if os.name == 'nt' else 'clear')
	train_batches = ImageDataGenerator().flow_from_directory(train_path,shuffle=False,target_size=target_size,classes=['ad','mci','nc'],batch_size=36)
	valid_batches = ImageDataGenerator().flow_from_directory(valid_path,shuffle=False,target_size=target_size,classes=['ad','mci','nc'],batch_size=24)
	test_batches = ImageDataGenerator().flow_from_directory(test_path,shuffle=False,target_size=target_size,classes=['ad','mci','nc'],batch_size=24)

	#imgs,labels = next(train_batches)
	#plots(imgs,titles=labels)
	model = Sequential([
		Conv2D(16, kernel_size=(3, 3),padding='same',input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)),
		BatchNormalization(),
		Activation('relu'),
		Conv2D(16, kernel_size=(3, 3),padding='same'),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Conv2D(32, kernel_size=(3, 3),padding='same'),
		Activation('relu'),
		Conv2D(32, kernel_size=(3, 3),padding='same',trainable=False),
		Activation('relu'),
		Conv2D(32, kernel_size=(3, 3),padding='same',trainable=False),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Conv2D(50, kernel_size=(3, 3),padding='same'),
		Activation('relu'),
		Conv2D(50, kernel_size=(3, 3),padding='same',trainable=False),
		Activation('relu'),
		Conv2D(50, kernel_size=(3, 3),padding='same',trainable=False),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Conv2D(100, kernel_size=(3, 3),padding='same'),
		Activation('relu'),
		Conv2D(100, kernel_size=(3, 3),padding='same',trainable=False),
		Activation('relu'),
		Conv2D(100, kernel_size=(3, 3),padding='same',trainable=False),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Flatten(),
		Dense(300,activation='softmax',trainable=False),
		Dense(100,activation='softmax',trainable=False),
		Dense(3, activation='softmax')
	])

	print(DATA_SET_EXECUTE[1])


	#vgg16 = keras.applications.vgg16.VGG16()
	#model = Sequential()
	#for layer in vgg16.layers:
		#model.add(layer)

	#model.layers.pop()
	#for layer in model.layers:
		#layer.trainable = False

	#model.add(Dense(3,activation='softmax'))
	print("Resumo do modelo")
	print(model.summary())

	#Criacao de imagem contendo o modelo utilizado
	#plot_model(model,show_shapes=True,to_file='model.png')

	#Compilacao da rede
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy','mse'])
	
	#Etapa de treinamento
	history = model.fit_generator(train_batches,steps_per_epoch=100,validation_data=valid_batches,validation_steps=50,epochs=NUMERO_EPOCHS,verbose=1)
	
	f = open('plots/history.txt','w')
	#Criacao do grafico com epochs do treinamento
	f.write(str(history.history['acc']))
	f.close()
	#plt.plot(history.history['acc'])
	#plt.title('Acuracia no treinamento')
	#plt.ylabel('Acuracia - %')
	#plt.xlabel('Epoch')

	#plt.savefig('history.png')
	
	#Etapa de teste
	pred = model.predict_generator(test_batches,verbose=1)
	pred = np.argmax(pred,axis=1)
	pred = np.expand_dims(pred,axis=1)
	pred = pred.reshape(1200,)
	error = np.sum(np.not_equal(pred,test_batches.classes)) / test_batches.samples

	f = open('plots/error.txt','w')
	#Criacao do grafico com epochs do treinamento
	f.write(str(error)+"%")
	f.close()

	## Compute confusion matrix
	cnf_matrix = confusion_matrix(test_batches.classes, pred)
	#np.set_printoptions(precision=2)
	f = open('plots/cm.txt','w')
	#Criacao do grafico com epochs do treinamento
	f.write(str(cnf_matrix))
	f.close()

	# Plot non-normalized confusion matrix
	#plt.figure()
	#plot_confusion_matrix(cnf_matrix, classes=['ad','mci','nc'],title='Confusion matrix, without normalization')
	#plt.savefig('confusion.png')
	# Plot normalized confusion matrix
	#plt.figure()
	#plot_confusion_matrix(cnf_matrix, classes=['ad','mci','nc'], normalize=True,title='Normalized confusion matrix')
	#plt.savefig('confusion_normalized.png')
	#error = np.sum(np.not_equal(pred,y_test)) / y_test.shape[0]
	#print("Taxa de erro: " + error + "%")
