from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

f = open("cm.txt","r")
values = f.read().split(";")
contador = 0
lista1 = []
lista2 = []
lista3 = []
for v in values:
	if contador < 3:
		lista1.append(int(v))
	elif contador < 6:
		lista2.append(int(v))
	else:
		lista3.append(int(v))
	contador += 1

lista_final = []
lista_final.append(lista1)
lista_final.append(lista2)
lista_final.append(lista3)

cm = np.array(lista_final)

print(cm)

classes = ['AD','MCI','NC']
normalize = False
title = 'Matriz de Confusão'
cmap = plt.cm.Blues

if normalize:
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print("Normalized confusion matrix")
else:
	print('Confusion matrix, without normalization')

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0)
plt.yticks(tick_marks, classes)

fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('Classes Reais')
plt.xlabel('Classes Preditadas')
plt.waitforbuttonpress()