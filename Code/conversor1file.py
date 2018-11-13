import os
import string
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn import plotting
from functs import *
import matplotlib.pyplot as plt
import cv2
from random import shuffle
import ipdb

CANNY = False
THRESHOLD_LOW = 80
THRESHOLD_HIGH = 150

CROPPED = False
RANGE = 40

tamanho_imagem = (166,256)

tamanho_final = (224,224)

larguraInicio = 0
alturaInicio = 0
alturaFim = 150

def getStartEnd(size):
	if size == 256:
		return (108,148)
	else:
		return (85,125)

diretorioOut = 'SaidaTemp/'
file = 'paciente1.nii'
arquivo = nib.load(file)
image = arquivo.get_data()
start,end = getStartEnd(image.shape[1])
segmentos = image[:,:,start:end]
iterador = 0
for i in range(RANGE):
	plt.imsave(diretorioOut+'z-img'+ str(iterador)+'.jpg',arr=segmentos[:,:,i],cmap='gray')
	imageSaved = cv2.imread(diretorioOut+'img'+ str(iterador)+'.jpg',0)
	resized_image = cv2.resize(imageSaved, tamanho_imagem)
	if CROPPED:
		altura, largura = resized_image.shape
		alturaMeio = altura // 2
		larguraMeio = largura // 2
		larguraInicio = 0
		larguraFim = 0
		alturaInicio = 0
		for j in range(largura):
			if resized_image[alturaMeio,j] > 0:
				larguraInicio = j
				break
		for j in reversed(range(largura)):
			if resized_image[alturaMeio,j] > 0:
				larguraFim = j
				break
		for i in range(altura):
			if resized_image[i,larguraMeio] > 0:
				alturaInicio = i
				break
		alturaInicio += 10
		larguraInicio += 10
		larguraFim -= 10
		CROPPED = True
		cropped = resized_image[alturaInicio:alturaInicio+alturaFim,larguraInicio:larguraInicio+larguraFim]
		resized_image_final = cv2.resize(cropped, tamanho_final)
	if CANNY and CROPPED:
		edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
		cv2.imwrite(diretorioOut+'img'+ str(iterador)+'.jpg',edges)
	else:
		cv2.imwrite(diretorioOut+'img'+ str(iterador)+'.jpg',resized_image)
	iterador += 1