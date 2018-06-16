import os
import string
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn import plotting
from functs import *
import matplotlib.pyplot as plt

#Informacoes para elaborar o PATH dos arquivos
classes = ['AD','CN','MCI']
base = 'ADNI'
tipo = 'MR'
formato = '.nii'
iterador = 0

#Diretorios
#Images/classe/subject/description/
#Arquivos
#ADNI_subject_MR_description.nii

(listaSubjectsAD,listaDescriptionAD,listaSubjectsMCI,listaDescriptionMCI,listaSubjectsCN,listaDescriptionCN) = obterSubjectsDescription()

for subject, description in zip(listaSubjectsAD,listaDescriptionAD):
	folder = '../Images/'+classes[0]+'/'+subject+'/'+formatarDescription(description)+'/'
	file = base+'_'+subject+'_'+tipo+'_'+formatarDescription(description)+formato
	arquivo = nib.load(folder+file)
	image = arquivo.get_data()
	segmentos = image[:,(image.shape[1] // 2):(image.shape[1] // 2)+50,:]
	diretorio = '../Output/Images/'+classes[0]+'/'+subject+'/'
	if not os.path.exists(diretorio):
		os.makedirs(diretorio)
	for i in range(50):
		plt.imsave(diretorio+'img'+str(iterador)+'.png',arr=segmentos[:,i,:],cmap='gray')
		iterador += 1
	iterador = 0
	#plt.imshow(segmento,cmap='gray')
	#plt.title('Imagem - ' + str(iterador))
	#iterador += 1
	#plt.waitforbuttonpress()

#formatarDescription(listaDescriptionAD[0]))

#print("Conversao de imagens NII para PNG")
#img = nib.load('002_s_0938_ad.nii')
#image = img.get_data()
#print(img.header)

#print(image.shape)

#print (image.shape[1] // 2)
######slices = image[:,(image.shape[1] // 2),:]
#soma = sum(sum(slice_1))
#linhas,colunas = slice_1.shape
#print(soma/(linhas*colunas))
#print(slice_1.shape)
#print(slices.shape)
#plt.imshow(slices,cmap='gray')
#plt.show()