import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras import optimizers
from keras import callbacks
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

config = tf.ConfigProto(intra_op_parallelism_threads=24,inter_op_parallelism_threads=2, allow_soft_placement=True,device_count = {'CPU': 24 })
session = tf.Session(config=config)
K.set_session(session)
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

train_path = 'dataset/train'
valid_path = 'dataset/valid'
test_path = 'dataset/test'
nome_arquivo = 'vgg16.txt'
DATA_SET_EXECUTE = ['ORIGINAL','CROPPED_WITHOUT_EDGE','CROPPED_EDGE']
NUMERO_EPOCHS = 40
input_shape = (224,224,3)
target_size = (224,224)

os.system('cls' if os.name == 'nt' else 'clear')
train_batches = ImageDataGenerator(rescale=1. / 255).flow_from_directory(train_path,target_size=target_size,classes=['ad','mci','nc'],batch_size=32)
valid_batches = ImageDataGenerator(rescale=1. / 255).flow_from_directory(valid_path,target_size=target_size,classes=['ad','mci','nc'],batch_size=12)
test_batches = ImageDataGenerator(rescale=1. / 255).flow_from_directory(test_path,target_size=target_size,classes=['ad','mci','nc'],batch_size=5)

print(DATA_SET_EXECUTE[1])

vgg16 = keras.applications.vgg16.VGG16(include_top=False,pooling=None)
model = Sequential()
contador = 0
divisor = [4,7,11,15,19]
for layer in vgg16.layers:
	model.add(layer)
	contador += 1
	if contador == divisor[0]:
		model.add(Dropout(0.25))
	elif contador == divisor[1]:
		model.add(Dropout(0.25))
	elif contador == divisor[2]:
		model.add(Dropout(0.25))
	elif contador == divisor[3]:
		model.add(Dropout(0.4))
	elif contador == divisor[4]:
		model.add(Dropout(0.4))

contador = 0
for layer in model.layers:
	contador += 1
	if contador not in divisor:
		layer.trainable = False

model.add(Flatten())
model.add(Dense(6,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))
print("Resumo do modelo")
print(model.summary())

#Criacao de imagem contendo o modelo utilizado
#plot_model(model,show_shapes=True,to_file='model.png')

#Compilacao da rede
model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.SGD(lr=0.032,nesterov=True),
          metrics=['accuracy','mse'])

#Etapa de treinamento
history = model.fit_generator(train_batches,steps_per_epoch=80,validation_data=valid_batches,validation_steps=40,epochs=NUMERO_EPOCHS,verbose=1)

f = open('logs-run/history' + nome_arquivo,'w')
#Criacao do grafico com epochs do treinamento
f.write(str(history.history['acc']))
f.close()

#Etapa de teste
pred = model.predict_generator(test_batches,verbose=1)
pred = np.argmax(pred,axis=1)
pred = np.expand_dims(pred,axis=1)
pred = pred.reshape(320,)
error = np.sum(np.not_equal(pred,test_batches.classes)) / test_batches.samples

f = open('logs-run/error' + nome_arquivo,'w')
#Criacao do grafico com epochs do treinamento
f.write(str(error)+"%")
f.close()

## Compute confusion matrix
cnf_matrix = confusion_matrix(test_batches.classes, pred)
f = open('logs-run/cm' + nome_arquivo,'w')
f.write(str(cnf_matrix))
f.close()
