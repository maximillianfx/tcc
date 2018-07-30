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
iteradorPacientes= 0
iteradorFolders = 1

larguraInicio = 0
alturaInicio = 0
alturaFim = 150

#nao utilizar loop para cada imagem, somente para a primeira e diminuir o +10-10 para algo

#256
#START = 108
#FIM = 148

#192
#START = 85
#FIM = 125


def getStartEnd(size):
	if size == 256:
		return (108,148)
	else:
		return (85,125)


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

tamanho_imagem = (166,256)

tamanho_final = (224,224)

CANNY = True
THRESHOLD_LOW = 80
THRESHOLD_HIGH = 150

#Diretorios
#Images/classe/subject/description/
#Arquivos
#ADNI_subject_MR_description.nii
qtd_192 = 0
lista_192 = []
qtd_256 = 0
lista_256 = []
outros = []

DEBUG = False
CROPPED = False
RANGE = 40

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
	CROPPED = False
	paciente = listaSubjectsAD[itemListaAD_Train]
	description = listaDescriptionAD[itemListaAD_Train]
	folder = '../Images/'+classes[0]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	if image.shape[1] == 256:
		qtd_256 += 1
		lista_256.append(paciente+str(iteradorPacientes))
	elif image.shape[1] == 192:
		qtd_192 += 1
		lista_192.append(paciente+str(iteradorPacientes))
	else:
		outros.append(paciente+str(iteradorPacientes))

	iteradorPacientes += 1

	start,end = getStartEnd(image.shape[1])
	segmentos = image[:,start:end,:]
	for i in range(RANGE):
		plt.imsave(diretorioTrain+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTrain+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		if not CROPPED:
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
		if CANNY:
			edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
			cv2.imwrite(diretorioTrain+'img'+ str(iterador)+'.jpg',edges)
		else:
			cv2.imwrite(diretorioTrain+'img'+ str(iterador)+'.jpg',resized_image_final)
		iterador += 1
for itemListaAD_Valid in seq_ad_valid:
	CROPPED = False
	paciente = listaSubjectsAD[itemListaAD_Valid]
	description = listaDescriptionAD[itemListaAD_Valid]
	folder = '../Images/'+classes[0]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	if image.shape[1] == 256:
		qtd_256 += 1
		lista_256.append(paciente+str(iteradorPacientes))
	elif image.shape[1] == 192:
		qtd_192 += 1
		lista_192.append(paciente+str(iteradorPacientes))
	else:
		outros.append(paciente+str(iteradorPacientes))

	iteradorPacientes += 1

	start,end = getStartEnd(image.shape[1])
	segmentos = image[:,start:end,:]
	for i in range(RANGE):
		plt.imsave(diretorioValid+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioValid+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		if not CROPPED:
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
		if CANNY:
			edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
			cv2.imwrite(diretorioValid+'img'+ str(iterador)+'.jpg',edges)
		else:
			cv2.imwrite(diretorioValid+'img'+ str(iterador)+'.jpg',resized_image_final)
		iterador += 1
for itemListaAD_Test in seq_ad_test:
	CROPPED = False
	paciente = listaSubjectsAD[itemListaAD_Test]
	description = listaDescriptionAD[itemListaAD_Test]
	folder = '../Images/'+classes[0]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	if image.shape[1] == 256:
		qtd_256 += 1
		lista_256.append(paciente+str(iteradorPacientes))
	elif image.shape[1] == 192:
		qtd_192 += 1
		lista_192.append(paciente+str(iteradorPacientes))
	else:
		outros.append(paciente+str(iteradorPacientes))

	iteradorPacientes += 1

	start,end = getStartEnd(image.shape[1])
	segmentos = image[:,start:end,:]
	for i in range(RANGE):
		plt.imsave(diretorioTest+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTest+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		if not CROPPED:
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
		if CANNY:
			edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
			cv2.imwrite(diretorioTest+'img'+ str(iterador)+'.jpg',edges)
		else:
			cv2.imwrite(diretorioTest+'img'+ str(iterador)+'.jpg',resized_image_final)
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
	CROPPED = False
	paciente = listaSubjectsCN[itemListaCN_Train]
	description = listaDescriptionCN[itemListaCN_Train]
	folder = '../Images/'+classes[1]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	if image.shape[1] == 256:
		qtd_256 += 1
		lista_256.append(paciente+str(iteradorPacientes))
	elif image.shape[1] == 192:
		qtd_192 += 1
		lista_192.append(paciente+str(iteradorPacientes))
	else:
		outros.append(paciente+str(iteradorPacientes))

	iteradorPacientes += 1

	start,end = getStartEnd(image.shape[1])
	segmentos = image[:,start:end,:]
	for i in range(RANGE):
		plt.imsave(diretorioTrain+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTrain+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		if not CROPPED:
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
		if CANNY:
			edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
			cv2.imwrite(diretorioTrain+'img'+ str(iterador)+'.jpg',edges)
		else:
			cv2.imwrite(diretorioTrain+'img'+ str(iterador)+'.jpg',resized_image_final)
		iterador += 1
for itemListaCN_Valid in seq_cn_valid:
	CROPPED = False
	paciente = listaSubjectsCN[itemListaCN_Valid]
	description = listaDescriptionCN[itemListaCN_Valid]
	folder = '../Images/'+classes[1]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	if image.shape[1] == 256:
		qtd_256 += 1
		lista_256.append(paciente+str(iteradorPacientes))
	elif image.shape[1] == 192:
		qtd_192 += 1
		lista_192.append(paciente+str(iteradorPacientes))
	else:
		outros.append(paciente+str(iteradorPacientes))

	iteradorPacientes += 1

	start,end = getStartEnd(image.shape[1])
	segmentos = image[:,start:end,:]
	for i in range(RANGE):
		plt.imsave(diretorioValid+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioValid+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		if not CROPPED:
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
		if CANNY:
			edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
			cv2.imwrite(diretorioValid+'img'+ str(iterador)+'.jpg',edges)
		else:
			cv2.imwrite(diretorioValid+'img'+ str(iterador)+'.jpg',resized_image_final)
		iterador += 1
for itemListaCN_Test in seq_cn_test:
	CROPPED = False
	paciente = listaSubjectsCN[itemListaCN_Test]
	description = listaDescriptionCN[itemListaCN_Test]
	folder = '../Images/'+classes[1]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	if image.shape[1] == 256:
		qtd_256 += 1
		lista_256.append(paciente+str(iteradorPacientes))
	elif image.shape[1] == 192:
		qtd_192 += 1
		lista_192.append(paciente+str(iteradorPacientes))
	else:
		outros.append(paciente+str(iteradorPacientes))

	iteradorPacientes += 1

	start,end = getStartEnd(image.shape[1])
	segmentos = image[:,start:end,:]
	for i in range(RANGE):
		plt.imsave(diretorioTest+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTest+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		if not CROPPED:
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
		if CANNY:
			edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
			cv2.imwrite(diretorioTest+'img'+ str(iterador)+'.jpg',edges)
		else:
			cv2.imwrite(diretorioTest+'img'+ str(iterador)+'.jpg',resized_image_final)
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
	CROPPED = False
	paciente = listaSubjectsMCI[itemListaMCI_Train]
	description = listaDescriptionMCI[itemListaMCI_Train]
	folder = '../Images/'+classes[2]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	if image.shape[1] == 256:
		qtd_256 += 1
		lista_256.append(paciente+str(iteradorPacientes))
	elif image.shape[1] == 192:
		qtd_192 += 1
		lista_192.append(paciente+str(iteradorPacientes))
	else:
		outros.append(paciente+str(iteradorPacientes))

	iteradorPacientes += 1

	start,end = getStartEnd(image.shape[1])
	segmentos = image[:,start:end,:]
	for i in range(RANGE):
		plt.imsave(diretorioTrain+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTrain+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		if not CROPPED:
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
		if CANNY:
			edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
			cv2.imwrite(diretorioTrain+'img'+ str(iterador)+'.jpg',edges)
		else:
			cv2.imwrite(diretorioTrain+'img'+ str(iterador)+'.jpg',resized_image_final)
		iterador += 1
for itemListaMCI_Valid in seq_mci_valid:
	CROPPED = False
	paciente = listaSubjectsMCI[itemListaMCI_Valid]
	description = listaDescriptionMCI[itemListaMCI_Valid]
	folder = '../Images/'+classes[2]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	if image.shape[1] == 256:
		qtd_256 += 1
		lista_256.append(paciente+str(iteradorPacientes))
	elif image.shape[1] == 192:
		qtd_192 += 1
		lista_192.append(paciente+str(iteradorPacientes))
	else:
		outros.append(paciente+str(iteradorPacientes))

	iteradorPacientes += 1

	start,end = getStartEnd(image.shape[1])
	segmentos = image[:,start:end,:]
	for i in range(RANGE):
		plt.imsave(diretorioValid+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioValid+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		if not CROPPED:
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
		if CANNY:
			edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
			cv2.imwrite(diretorioValid+'img'+ str(iterador)+'.jpg',edges)
		else:
			cv2.imwrite(diretorioValid+'img'+ str(iterador)+'.jpg',resized_image_final)
		iterador += 1
for itemListaMCI_Test in seq_mci_test:
	CROPPED = False
	paciente = listaSubjectsMCI[itemListaMCI_Test]
	description = listaDescriptionMCI[itemListaMCI_Test]
	folder = '../Images/'+classes[2]+'/'+paciente+'/'+formatarDescription(description)+'/'
	file = base+'_'+paciente+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	if image.shape[1] == 256:
		qtd_256 += 1
		lista_256.append(paciente+str(iteradorPacientes))
	elif image.shape[1] == 192:
		qtd_192 += 1
		lista_192.append(paciente+str(iteradorPacientes))
	else:
		outros.append(paciente+str(iteradorPacientes))

	iteradorPacientes += 1

	start,end = getStartEnd(image.shape[1])
	segmentos = image[:,start:end,:]
	for i in range(RANGE):
		plt.imsave(diretorioTest+'img'+ str(iterador)+'.jpg',arr=segmentos[:,i,:],cmap='gray')
		imageSaved = cv2.imread(diretorioTest+'img'+ str(iterador)+'.jpg',0)
		resized_image = cv2.resize(imageSaved, tamanho_imagem)
		if not CROPPED:
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
		if CANNY:
			edges = cv2.Canny(resized_image_final,THRESHOLD_LOW,THRESHOLD_HIGH)
			cv2.imwrite(diretorioTest+'img'+ str(iterador)+'.jpg',edges)
		else:
			cv2.imwrite(diretorioTest+'img'+ str(iterador)+'.jpg',resized_image_final)
		iterador += 1

print('Conversoes finalizadas!')
if DEBUG:
	import ipdb
	ipdb.set_trace()
	print('Quantidade 256 = ' + str(qtd_256))
	print('Lista 256:')
	print(lista_256)
	print('Quantidade 192 = ' + str(qtd_192))
	print('Lista 192:')
	print(lista_192)
	print('Outros:')
	print(outros)
	#plt.imshow(segmento,cmap='gray')
	#plt.title('Imagem - ' + str(iterador))
	#iterador += 1
	#plt.waitforbuttonpress()
