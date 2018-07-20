import string
import os
import cv2
import ipdb

image = cv2.imread('img-test2.jpg',0)

#ipdb.set_trace()

altura, largura = image.shape

alturaMeio = altura // 2
larguraMeio = largura // 2
larguraInicio = 0
alturaInicio = 0
larguraFim = 0

alturaFim = 150

for j in range(largura):
	if image[alturaMeio,j] > 0:
		larguraInicio = j
		break
print(larguraInicio)

for j in reversed(range(largura)):
	if image[alturaMeio,j] > 0:
		larguraFim = j
		break

print(larguraFim)

for i in range(altura):
	if image[i,larguraMeio] > 0:
		alturaInicio = i
		break


print(alturaInicio)

alturaInicio += 10
larguraInicio += 10
larguraFim -= 10

cropped = image[alturaInicio:alturaInicio+alturaFim,larguraInicio:larguraInicio+larguraFim]

cv2.imshow('Janela',cropped)
cv2.imshow('JanelaNormal',image)
cv2.waitKey(0)
cv2.destroyAllWindows()