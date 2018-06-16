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


img = cv.imread(diretorio+classes[0]+'/'+listaSubjectsAD[0]+'/img0.png',0)
blur = cv.medianBlur(img,3)
alpha = 1.2
beta = 0
contraste = cv.addWeighted(blur,alpha,np.zeros(blur.shape,blur.dtype),0,beta)
cv.namedWindow('AD', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur', cv.WINDOW_NORMAL)
cv.namedWindow('AD-Blur-Contraste', cv.WINDOW_NORMAL)
cv.imshow('AD',img)
cv.imshow('AD-Blur',blur)
cv.imshow('AD-Blur-Contraste',contraste)
cv.waitKey(0)
cv.destroyAllWindows()