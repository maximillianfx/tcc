import os
import string
import csv

def obterSubjectsDescription():
	files = ['Images/Alzheimer_Disease_6_12_2018.csv','Images/Cognitively_Normal_6_06_2018.csv','Images/Mild_Cognitive_Impairment_6_08_2018.csv']

	listaSubjectsAD = []
	listaDescriptionAD = []
	listaSubjectsCN = []
	listaDescriptionCN =[]
	listaSubjectsMCI = []
	listaDescriptionMCI= []
	iterador = 0

	for file in files:
		with open(file) as csvFile:
			reader = csv.DictReader(csvFile)
			for row in reader:
				if iterador == 0:
					listaSubjectsAD.append(row['Subject'])
					listaDescriptionAD.append(row['Description'])
				elif iterador == 1:
					listaSubjectsCN.append(row['Subject'])
					listaDescriptionCN.append(row['Description'])
				else:
					listaSubjectsMCI.append(row['Subject'])
					listaDescriptionMCI.append(row['Description'])
			iterador += 1
		
	return (listaSubjectsAD,listaDescriptionAD,listaSubjectsMCI,listaDescriptionMCI,listaSubjectsCN,listaDescriptionCN)

def formatarDescription(description):
	description_splitted = description.split("; ")
	correction = description_splitted[2].split(" ")
	return description_splitted[0]+"__"+description_splitted[1]+"__"+correction[0]+"_"+correction[1]+"__"+description_splitted[3]