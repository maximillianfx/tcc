import os
import string
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn import plotting
import matplotlib.pyplot as plt
import cv2
from random import shuffle
import ipdb

classes = ['AD','CN','MCI']
classes_lowerCase = ['ad','nc','mci']

contadorGeral = 0

tamanho_final = (224,224)

diretorioAD_test = 'ds-3c-canny/test/ad/'
diretorioNC_test = 'ds-3c-canny/test/nc/'
diretorioMCI_test = 'ds-3c-canny/test/mci/'

diretorioAD_valid = 'ds-3c-canny/valid/ad/'
diretorioNC_valid = 'ds-3c-canny/valid/nc/'
diretorioMCI_valid = 'ds-3c-canny/valid/mci/'

diretorioAD_train = 'ds-3c-canny/train/ad/'
diretorioNC_train = 'ds-3c-canny/train/nc/'
diretorioMCI_train = 'ds-3c-canny/train/mci/'

THRESHOLD_LOW = 40
THRESHOLD_HIGH = 100

print("Aplicando Canny")
for c in range(1,61):
	img = cv2.imread(diretorioAD_test+str(c)+'img.jpg',0)
	edges = cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH)
	cv2.imwrite(diretorioAD_test+str(c)+'img.jpg',edges)
	contadorGeral += 1

for c in range(61,101):
	img = cv2.imread(diretorioMCI_test+str(c)+'img.jpg',0)
	edges = cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH)
	cv2.imwrite(diretorioMCI_test+str(c)+'img.jpg',edges)
	contadorGeral += 1

for c in range(101,161):
	img = cv2.imread(diretorioNC_test+str(c)+'img.jpg',0)
	edges = cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH)
	cv2.imwrite(diretorioNC_test+str(c)+'img.jpg',edges)
	contadorGeral += 1

for c in range(161,341):
	img = cv2.imread(diretorioAD_valid+str(c)+'img.jpg',0)
	edges = cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH)
	cv2.imwrite(diretorioAD_valid+str(c)+'img.jpg',edges)
	contadorGeral += 1

for c in range(341,541):
	img = cv2.imread(diretorioMCI_valid+str(c)+'img.jpg',0)
	edges = cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH)
	cv2.imwrite(diretorioMCI_valid+str(c)+'img.jpg',edges)
	contadorGeral += 1

for c in range(541,701):
	img = cv2.imread(diretorioNC_valid+str(c)+'img.jpg',0)
	edges = cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH)
	cv2.imwrite(diretorioNC_valid+str(c)+'img.jpg',edges)
	contadorGeral += 1

for c in range(701,1701):
	img = cv2.imread(diretorioAD_train+str(c)+'img.jpg',0)
	edges = cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH)
	cv2.imwrite(diretorioAD_train+str(c)+'img.jpg',edges)
	contadorGeral += 1

for c in range(1701,2701):
	img = cv2.imread(diretorioMCI_train+str(c)+'img.jpg',0)
	edges = cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH)
	cv2.imwrite(diretorioMCI_train+str(c)+'img.jpg',edges)
	contadorGeral += 1

for c in range(2701,3661):
	img = cv2.imread(diretorioNC_train+str(c)+'img.jpg',0)
	edges = cv2.Canny(img,THRESHOLD_LOW,THRESHOLD_HIGH)
	cv2.imwrite(diretorioNC_train+str(c)+'img.jpg',edges)
	contadorGeral += 1


print("Processamento finalizado")