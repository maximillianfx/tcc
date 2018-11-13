import string
import os
import cv2
import ipdb
import numpy as np

image = cv2.imread('img-gamma-aplicacao.png',0)
#blur = cv2.medianBlur(image,3)
altura, largura = image.shape
#blank_image = np.zeros((altura,largura,1), np.uint8)

#for i in range(altura):
#	for j in range(largura):
#		blank_image[i,j] = (((blur[i,j]/255)**(1.3))*(255))

#cv2.imwrite('teste-blur-real.png',blur)
#exit(1)


alturaMeio = altura // 2
larguraMeio = largura // 2
larguraInicio = 0
alturaInicio = 0
larguraFim = 0

alturaFim = 180

for j in range(largura):
	if image[alturaMeio,j] > 55:
		larguraInicio = j
		break
print(larguraInicio)

for j in reversed(range(largura)):
	if image[alturaMeio,j] > 55:
		larguraFim = j
		break

print(larguraFim)

for i in range(altura):
	if image[i,larguraMeio] > 55:
		alturaInicio = i
		break


print(alturaInicio)

alturaInicio += 10
larguraInicio += 10
larguraFim -= 10

cropped = image[alturaInicio:alturaFim,larguraInicio:larguraFim]

cv2.imshow('Janela',cropped)
cv2.imshow('JanelaNormal',image)
cv2.waitKey(0)
cv2.destroyAllWindows()