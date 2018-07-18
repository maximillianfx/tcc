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

#Informacoes para elaborar o PATH dos arquivos
classes = ['AD','CN','MCI']
classes_lowerCase = ['ad','nc','mci']
base = 'ADNI'
tipo = 'MR'
formato = '.nii'
iterador = 0
iteradorFolders = 1

sequenciaAD = [i for i in range(0,50)]
sequenciaCN = [i for i in range(0,50)]
sequenciaMCI = [i for i in range(0,50)]

shuffle(sequenciaAD)
shuffle(sequenciaCN)
shuffle(sequenciaMCI)

seq_ad_train = sequenciaAD[0:30]
seq_cn_train = sequenciaCN[0:30]
seq_mci_train = sequenciaMCI[0:30]

seq_ad_valid = sequenciaAD[30:40]
seq_cn_valid = sequenciaCN[30:40]
seq_mci_valid = sequenciaMCI[30:40]

seq_ad_test = sequenciaAD[40:50]
seq_cn_test = sequenciaCN[40:50]
seq_mci_test = sequenciaMCI[40:50]

tamanho_imagem = (224,224)


#Diretorios
#Images/classe/subject/description/
#Arquivos
#ADNI_subject_MR_description.nii

print('Conversor de arquivos .nii para JPG')
(listaSubjectsAD,listaDescriptionAD,listaSubjectsMCI,listaDescriptionMCI,listaSubjectsCN,listaDescriptionCN) = obterSubjectsDescription()

print('Convertendo arquivos de pacientes AD...')
diretorioTrain = '../Output/dataset/train/' + classes_lowerCase[0] + '/'
diretorioValid = '../Output/dataset/valid/' + classes_lowerCase[0] + '/'
diretorioTest = '../Output/dataset/test/' + classes_lowerCase[0] + '/'
if not (os.path.exists(diretorioTrain) or os.path.exists(diretorioValid) or os.path.exists(diretorioTest)):
	os.makedirs(diretorioTrain)
	os.makedirs(diretorioValid)
	os.makedirs(diretorioTest)
for itemListaAD_Train in seq_ad_train:
	paciente = listaSubjectsAD[itemListaAD_Train]
	description = listaDescriptionAD[itemListaAD_Train]
	folder = '../Images/'+classes[0]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	#import ipdb
	#ipdb.set_trace()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	for i in range(50):
		plt.imsave(diretorioTrain+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTrain+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		cv2.imwrite(diretorioTrain+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1
for itemListaAD_Valid in seq_ad_valid:
	paciente = listaSubjectsAD[itemListaAD_Valid]
	description = listaDescriptionAD[itemListaAD_Valid]
	folder = '../Images/'+classes[0]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	for i in range(50):
		plt.imsave(diretorioValid+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioValid+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		cv2.imwrite(diretorioValid+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1
for itemListaAD_Test in seq_ad_test:
	paciente = listaSubjectsAD[itemListaAD_Test]
	description = listaDescriptionAD[itemListaAD_Test]
	folder = '../Images/'+classes[0]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	for i in range(50):
		plt.imsave(diretorioTest+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTest+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		cv2.imwrite(diretorioTest+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1

print('Conversao OK')
print('Convertendo arquivos de pacientes CN...')
diretorioTrain = '../Output/dataset/train/' + classes_lowerCase[1] + '/'
diretorioValid = '../Output/dataset/valid/' + classes_lowerCase[1] + '/'
diretorioTest = '../Output/dataset/test/' + classes_lowerCase[1] + '/'
if not (os.path.exists(diretorioTrain) or os.path.exists(diretorioValid) or os.path.exists(diretorioTest)):
	os.makedirs(diretorioTrain)
	os.makedirs(diretorioValid)
	os.makedirs(diretorioTest)
for itemListaCN_Train in seq_cn_train:
	paciente = listaSubjectsCN[itemListaCN_Train]
	description = listaDescriptionCN[itemListaCN_Train]
	folder = '../Images/'+classes[1]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	for i in range(50):
		plt.imsave(diretorioTrain+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTrain+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		cv2.imwrite(diretorioTrain+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1
for itemListaCN_Valid in seq_cn_valid:
	paciente = listaSubjectsCN[itemListaCN_Valid]
	description = listaDescriptionCN[itemListaCN_Valid]
	folder = '../Images/'+classes[1]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	for i in range(50):
		plt.imsave(diretorioValid+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioValid+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		cv2.imwrite(diretorioValid+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1
for itemListaCN_Test in seq_cn_test:
	paciente = listaSubjectsCN[itemListaCN_Test]
	description = listaDescriptionCN[itemListaCN_Test]
	folder = '../Images/'+classes[1]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	for i in range(50):
		plt.imsave(diretorioTest+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTest+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		cv2.imwrite(diretorioTest+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1

print('Conversao OK')
print('Convertendo arquivos de pacientes MCI...')
diretorioTrain = '../Output/dataset/train/' + classes_lowerCase[2] + '/'
diretorioValid = '../Output/dataset/valid/' + classes_lowerCase[2] + '/'
diretorioTest = '../Output/dataset/test/' + classes_lowerCase[2] + '/'
if not (os.path.exists(diretorioTrain) or os.path.exists(diretorioValid) or os.path.exists(diretorioTest)):
	os.makedirs(diretorioTrain)
	os.makedirs(diretorioValid)
	os.makedirs(diretorioTest)
for itemListaMCI_Train in seq_mci_train:
	paciente = listaSubjectsMCI[itemListaMCI_Train]
	description = listaDescriptionMCI[itemListaMCI_Train]
	folder = '../Images/'+classes[2]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	for i in range(50):
		plt.imsave(diretorioTrain+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTrain+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		cv2.imwrite(diretorioTrain+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1
for itemListaMCI_Valid in seq_mci_valid:
	paciente = listaSubjectsMCI[itemListaMCI_Valid]
	description = listaDescriptionMCI[itemListaMCI_Valid]
	folder = '../Images/'+classes[2]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	for i in range(50):
		plt.imsave(diretorioValid+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioValid+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		cv2.imwrite(diretorioValid+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1
for itemListaMCI_Test in seq_mci_test:
	paciente = listaSubjectsMCI[itemListaMCI_Test]
	description = listaDescriptionMCI[itemListaMCI_Test]
	folder = '../Images/'+classes[2]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	for i in range(50):
		plt.imsave(diretorioTest+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTest+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		cv2.imwrite(diretorioTest+'img'+ str(iterador)+'.jpg',resized_image)
		iterador += 1

print('Conversoes finalizadas!')
	#plt.imshow(segmento,cmap='gray')
	#plt.title('Imagem - ' + str(iterador))
	#iterador += 1
	#plt.waitforbuttonpress()