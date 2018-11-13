from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

f = open("history.txt","r")
r = open("history-val.txt","r")
values = f.read().split(";")
lista = []
for v in values:
	lista.append(float(v))

values_val = r.read().split(";")
lista_val = []
for v in values_val:
	lista_val.append(float(v))


plt.plot(lista,label='Treinamento')
plt.plot(lista_val,label='Validacao')
plt.legend(loc='upper left')
plt.title('Perda')
plt.xlabel('Epochs')
plt.ylabel('Perda - CE')
plt.waitforbuttonpress()