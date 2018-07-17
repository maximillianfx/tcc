import os
import string
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn import plotting
from functs import *
import matplotlib.pyplot as plt
import cv2

#Informacoes para elaborar o PATH dos arquivos
classes = ['AD','CN','MCI']
base = 'ADNI'
tipo = 'MR'
formato = '.nii'
iterador = 0
iteradorFolders = 1

#Diretorios
#Images/classe/subject/description/
#Arquivos
#ADNI_subject_MR_description.nii

print('Conversor de arquivos .nii para PNG')
(listaSubjectsAD,listaDescriptionAD,listaSubjectsMCI,listaDescriptionMCI,listaSubjectsCN,listaDescriptionCN) = obterSubjectsDescription()

print('Convertendo arquivos de pacientes AD...')
for subject, description in zip(listaSubjectsAD,listaDescriptionAD):
	folder = '../Images/'+classes[0]+'/'+subject+'/'+formatarDescription(description)+'/'
	file = base+'_'+subject+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	diretorio = '../Output/Images/' + classes[0]+ '/'+ str(iteradorFolders) + '-' + subject+'/'
	iteradorFolders += 1
	if not os.path.exists(diretorio):
		os.makedirs(diretorio)
	for i in range(50):
		plt.imsave(diretorio+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorio+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, (166, 256))
		cv2.imwrite(diretorio+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1
iteradorFolders = 0
print('Conversao OK')
print('Convertendo arquivos de pacientes CN...')
for subject, description in zip(listaSubjectsCN,listaDescriptionCN):
	folder = '../Images/'+classes[1]+'/'+subject+'/'+formatarDescription(description)+'/'
	file = base+'_'+subject+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	diretorio = '../Output/Images/' + classes[1]+ '/'+ str(iteradorFolders) + '-' + subject+'/'
	iteradorFolders += 1
	if not os.path.exists(diretorio):
		os.makedirs(diretorio)
	for i in range(50):
		plt.imsave(diretorio+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorio+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, (166, 256))
		cv2.imwrite(diretorio+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1
iteradorFolders = 0
print('Conversao OK')
print('Convertendo arquivos de pacientes MCI...')
for subject, description in zip(listaSubjectsMCI,listaDescriptionMCI):
	folder = '../Images/'+classes[2]+'/'+subject+'/'+formatarDescription(description)+'/'
	file = base+'_'+subject+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	diretorio = '../Output/Images/' + classes[2]+ '/'+ str(iteradorFolders) + '-' + subject+'/'
	iteradorFolders += 1
	if not os.path.exists(diretorio):
		os.makedirs(diretorio)
	for i in range(50):
		plt.imsave(diretorio+'img'+str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorio+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, (166, 256))
		cv2.imwrite(diretorio+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1

print('Conversoes finalizadas!')
	#plt.imshow(segmento,cmap='gray')
	#plt.title('Imagem - ' + str(iterador))
	#iterador += 1
	#plt.waitforbuttonpress()