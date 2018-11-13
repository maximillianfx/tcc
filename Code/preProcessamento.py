import os
import string
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn import plotting
from functs import *
import matplotlib.pyplot as plt
import cv2 as cv
from functs import *

#Informacoes para elaborar o PATH dos arquivos
classes = ['AD','CN','MCI']
diretorio = '../Output/Images/'
iterador = 0

#Diretorios
#Output/Images/subject/
#Arquivos
#imgX.png - X: numero de 0 a 49

(listaSubjectsAD,listaDescriptionAD,listaSubjectsMCI,listaDescriptionMCI,listaSubjectsCN,listaDescriptionCN) = obterSubjectsDescription()


img = cv.imread('img-test2.jpg',0)
altura, largura = img.shape
blur = cv.medianBlur(img,3)
blank_image1 = np.zeros((altura,largura,1), np.uint8)
blank_image2 = np.zeros((altura,largura,1), np.uint8)
blank_image3 = np.zeros((altura,largura,1), np.uint8)
blank_image4 = np.zeros((altura,largura,1), np.uint8)
blank_image5 = np.zeros((altura,largura,1), np.uint8)
blank_image6 = np.zeros((altura,largura,1), np.uint8)
blank_image7 = np.zeros((altura,largura,1), np.uint8)
blank_image8 = np.zeros((altura,largura,1), np.uint8)
blank_image9 = np.zeros((altura,largura,1), np.uint8)

#ipdb.set_trace()

for i in range(altura):
	for j in range(largura):
		blank_image1[i,j] = (((blur[i,j]/255)**(1.1))*(255))
		blank_image2[i,j] = (((blur[i,j]/255)**(1.2))*(255))
		blank_image3[i,j] = (((blur[i,j]/255)**(1.3))*(255))
		blank_image4[i,j] = (((blur[i,j]/255)**(1.4))*(255))
		blank_image5[i,j] = (((blur[i,j]/255)**(1.5))*(255))
		blank_image6[i,j] = (((blur[i,j]/255)**(1.6))*(255))
		blank_image7[i,j] = (((img[i,j]/255)**(0.7))*(255))
		blank_image8[i,j] = (((blur[i,j]/255)**(1.8))*(255))
		blank_image9[i,j] = (((blur[i,j]/255)**(1.9))*(255))


blur7 = cv.medianBlur(blank_image7,3)
cv.namedWindow('AD', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Adjust1', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Adjust2', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Adjust3', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Adjust4', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Adjust5', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Adjust6', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Adjust7', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Adjust8', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Adjust9', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur7', cv.WINDOW_NORMAL)
cv.imshow('AD',img)
cv.imshow('AD-Blur',blur)
cv.imshow('AD-Blur-Adjust1',blank_image1)
cv.imshow('AD-Blur-Adjust2',blank_image2)
cv.imshow('AD-Blur-Adjust3',blank_image3)
cv.imshow('AD-Blur-Adjust4',blank_image4)
cv.imshow('AD-Blur-Adjust5',blank_image5)
cv.imshow('AD-Blur-Adjust6',blank_image6)
cv.imshow('AD-Blur-Adjust7',blank_image7)
cv.imshow('AD-Blur-Adjust8',blank_image8)
cv.imshow('AD-Blur-Adjust9',blank_image9)
cv.imshow('AD-Blur7',blur7)
cv.waitKey(0)
cv.destroyAllWindows()