from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

f = open("history.txt","r")
values = f.read().split(";")
lista = []
for v in values:
	lista.append(float(v)*100)


plt.plot(lista)
plt.title('Acuracia')
plt.xlabel('Epochs')
plt.ylabel('Acuracia %')
plt.waitforbuttonpress()