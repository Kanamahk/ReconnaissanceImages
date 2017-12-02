#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors
import timeit


def getJustImages(data):
	if(data == "test"):
		return np.load("data/tst_img.npy")
	if(data == "dev"):
		return np.load("data/dev_img.npy")
	if(data == "training"):
		return np.load("data/trn_img.npy")


def lectureDonnneesTuple(data):
	if(data == "test"):
		X0 = np.load("data/tst_img.npy")
		lbl0 = np.load("data/tst_lbl.npy")
		return (X0, lbl0)

	if(data == "dev"):
		X0 = np.load("data/dev_img.npy")
		lbl0 = np.load("data/dev_lbl.npy")
		return (X0, lbl0)
		
	if(data == "training"):
		X0 = np.load("data/trn_img.npy")
		lbl0 = np.load("data/trn_lbl.npy")
		return (X0, lbl0)
	

def lectureDonnneesList(data):
	X0 = None
	lbl0 = None 
	if(data == "test"):
		X0 = np.load("data/tst_img.npy")
		lbl0 = np.load("data/tst_lbl.npy")

	if(data == "dev"):
		X0 = np.load("data/dev_img.npy")
		lbl0 = np.load("data/dev_lbl.npy")
		
	if(data == "training"):
		X0 = np.load("data/trn_img.npy")
		lbl0 = np.load("data/trn_lbl.npy")
	
	liste = []
	if(len(X0) == len(lbl0)):
			for i in range(0, len(X0)):
				liste.append((X0[i], lbl0[i]))

	return liste

def lectureDonnneesDictList(data):
	X0 = None
	lbl0 = None 
	if(data == "test"):
		X0 = np.load("data/tst_img.npy")
		lbl0 = np.load("data/tst_lbl.npy")

	if(data == "dev"):
		X0 = np.load("data/dev_img.npy")
		lbl0 = np.load("data/dev_lbl.npy")
		
	if(data == "training"):
		X0 = np.load("data/trn_img.npy")
		lbl0 = np.load("data/trn_lbl.npy")
	
	Dict = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6 : [], 7 : [], 8 : [], 9 : []}
	if(len(X0) == len(lbl0)):
			for i in range(0, len(X0)):
				Dict[lbl0[i]].append(X0[i])
	return Dict

def tupleDonnneesToDictList(data):
	Dict = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6 : [], 7 : [], 8 : [], 9 : []}
	if(len(data[0]) == len(data[1])):
			for i in range(0, len(data[0])):
				Dict[data[1][i]].append(data[0][i])
	return Dict

def afficheImage(image):
	createImage(image)
	plt.show()

def createImage(image):
	img = image.reshape(28,28)
	plt.imshow(img, plt.cm.gray)


def convertLbl(lbl):
	if(lbl == 0):
		return 'T-shirt/top'
	if(lbl == 1):
		return 'Trouser'
	if(lbl == 2):
		return 'Pullover'
	if(lbl == 3):
		return 'Dress'
	if(lbl == 4):
		return 'Coat'
	if(lbl == 5):
		return 'Sandal'
	if(lbl == 6):
		return 'Shirt'
	if(lbl == 7):
		return 'Sneaker'
	if(lbl == 8):
		return 'Bag'
	if(lbl == 9):
		return 'Ankle boot'

def createAverageImage(groupeImage):
	tailleImage = len(groupeImage[0])
	res = np.zeros(tailleImage)
	for i in range(0, tailleImage):
		for j in range(0, len(groupeImage)):		
			res[i] += groupeImage[j][i]
		res[i] /= len(groupeImage)
	return res;
			
def createAverageImages(Dict):
	res = {}
	for i in range(0, 10):
		res[i] =  createAverageImage(Dict[i])
	return res


def afficheAverageImages(Dict, size):
	fig, axes = plt.subplots(nrows=2, ncols=5)
	fig.tight_layout()
	sqrtSize = int(np.sqrt(size))
	for i in range(0, 10):
		line = int(i/5)
		row = i%5
		img = Dict[i].reshape(sqrtSize,sqrtSize)
		axes[line, row].imshow(img, plt.cm.gray)
		axes[line, row].set_title(convertLbl(i))

	plt.show()


def calcDifference(image, averageImage):
	res = 0
	for i in range(0, len(image)):
		temp = averageImage[i] - image[i]
		res += -temp if temp < 0 else temp
	return res

def calcAllDifference(image, averageImages):
	total = 0
	tailleDuPlusPetit = 255*len(image) #difference max
	plusPetit= 0;
	Liste = []
	for i in range(0, len(averageImages)):
		Liste.append(calcDifference(image, averageImages[i]))
		if Liste[i] < tailleDuPlusPetit :
			tailleDuPlusPetit = Liste[i]
			plusPetit = i
		total += Liste[i]
	return (plusPetit, Liste, total)

def makeAGuessDMIN(data, averageImages, string, Dict):
	difs = calcAllDifference(data[0], averageImages)
	#print('Je pense que ce vetement est un ' + convertLbl(difs[0]) + '.')
	string = string + 'Je pense que ce vetement est un ' + convertLbl(difs[0]) + '.\n'

	if difs[0] == data[1]:
		#print('Ce vetement est effectivement un ' + convertLbl(difs[0]) + '.')
		string = string + 'Ce vetement est effectivement un ' + convertLbl(difs[0]) + '.\n\n'
		
		Dict[difs[0]]['nbFound'] += 1
		res = 1
	else:
		#print('Ce vetement etait en fait un ' + convertLbl(data[1]) + '.')
		string = string + 'Ce vetement etait en fait un ' + convertLbl(data[1]) + '.\n\n'
		res = 0

	Dict[data[1]]['nbTotal'] += 1
	Dict[data[1]][difs[0]] += 1

	#print()
	return (res, string)

def makeAllGuessDMIN(datas, averageImages, name, tempsPreparation):
	nbReconnu = 0
	ConfusionMatrice =  {'TempsCalcul' : tempsPreparation,
						0 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						1 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						2 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						3 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						4 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						5 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						6 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						7 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						8 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						9 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0}}
	
	res = (0, "")

	
	start_time = timeit.default_timer()
	
	for i in range(0, len(datas[0])):
		res = makeAGuessDMIN((datas[0][i], datas[1][i]), averageImages, res[1], ConfusionMatrice)
		nbReconnu += res[0]
	
	ConfusionMatrice['TempsCalcul'] += timeit.default_timer() - start_time
	
	string = res[1]

	for i in range(0, len(ConfusionMatrice)-1):
		string = string + '\nJ\'ai reconnu ' + str(ConfusionMatrice[i]['nbFound']) + ' ' + convertLbl(i) +  ' sur ' + str(ConfusionMatrice[i]['nbTotal']) + ' (' + str(100 * ConfusionMatrice[i]['nbFound']/ConfusionMatrice[i]['nbTotal']) + '%).\n'
		for j in range(0, len(ConfusionMatrice)-1):
			if(i != j):
				string = string + 'J\'ai confondu ' + str(ConfusionMatrice[i][j]) + ' ' + convertLbl(j) + ' pour des ' + convertLbl(i) + '.\n'
		

	#print('\nJ\'ai reconnu ' + str(nbReconnu) + ' vetements sur ' + str(len(datas)) + ' (' + str(100 * nbReconnu/len(datas)) + '%).')
	string = string + '\nJ\'ai reconnu ' + str(nbReconnu) + ' vetements sur ' + str(len(datas[0])) + ' (' + str(100 * nbReconnu/len(datas[0])) + '%) en ' + str(ConfusionMatrice['TempsCalcul']) + ' secondes.'
	write_in_file(string, "resultats" + name +".txt")
	
	MatriceToString = ''
	for i in range(0, 10):
		for j in range(0, 10):
			MatriceToString = MatriceToString + str(ConfusionMatrice[i][j]) + '\t'
		MatriceToString = MatriceToString + '\n'	
	write_in_file(MatriceToString, "matrice" + name +".txt")
	
	return ConfusionMatrice

	
def makeAGuessLSVM_NNC(data, clf, string, Dict, classifieur):
	#difs = calcAllDifference(data[0], averageImages)
	prediction = clf.predict([data[0]])[0]

	string = string + 'Le classifieur ' + classifieur + ' pense que ce vetement est un ' + convertLbl(prediction) + '.\n'

	if prediction == data[1]:
		string = string + 'Ce vetement est effectivement un ' + convertLbl(prediction) + '.\n\n'
		
		Dict[prediction]['nbFound'] += 1
		res = 1
	else:
		string = string + 'Ce vetement etait en fait un ' + convertLbl(data[1]) + '.\n\n'
		res = 0

	Dict[data[1]]['nbTotal'] += 1
	Dict[data[1]][prediction] += 1

	return (res, string)	
	
def makeAllGuessLSVM_NNC(clf, datas, tempsPreparation, classifieur):
	nbReconnu = 0
	ConfusionMatrice =  {'TempsCalcul' : tempsPreparation,
						0 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						1 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						2 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						3 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						4 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						5 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						6 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						7 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						8 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0},
						9 : {'nbTotal' : 0, 'nbFound' : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0}}
	
	res = (0, "")
	
	print("Commencement de la reconnaissance.")
	start_time = timeit.default_timer()
	for i in range(0, len(datas[0])):
		res = makeAGuessLSVM_NNC((datas[0][i], datas[1][i]), clf, res[1], ConfusionMatrice, classifieur)
		nbReconnu += res[0]
	ConfusionMatrice['TempsCalcul'] += timeit.default_timer() - start_time
	print("Reconnaissance terminé.")
	
	string = res[1]

	for i in range(0, len(ConfusionMatrice)-1):
		string = string + '\nLe classifieur ' + classifieur + ' a reconnu ' + str(ConfusionMatrice[i]['nbFound']) + ' ' + convertLbl(i) +  ' sur ' + str(ConfusionMatrice[i]['nbTotal']) + ' (' + str(100 * ConfusionMatrice[i]['nbFound']/ConfusionMatrice[i]['nbTotal']) + '%).\n'
		for j in range(0, len(ConfusionMatrice)-1):
			if(i != j):
				string = string + 'Le classifieur ' + classifieur + ' a confondu ' + str(ConfusionMatrice[i][j]) + ' ' + convertLbl(j) + ' pour des ' + convertLbl(i) + '.\n'
				string = string + 'Le classifieur ' + classifieur + ' a confondu ' + str(ConfusionMatrice[i][j]) + ' ' + convertLbl(j) + ' pour des ' + convertLbl(i) + '.\n'
		
	string = string + '\nJ\'ai reconnu ' + str(nbReconnu) + ' vetements sur ' + str(len(datas[0])) + ' (' + str(100 * nbReconnu/len(datas[0])) + '%) en ' + str(ConfusionMatrice['TempsCalcul']) + ' secondes.'
	write_in_file(string, "resultats" + classifieur + ".txt")
	
	MatriceToString = ''
	for i in range(0, 10):
		for j in range(0, 10):
			MatriceToString = MatriceToString + str(ConfusionMatrice[i][j]) + '\t'
		MatriceToString = MatriceToString + '\n'	
	write_in_file(MatriceToString, "matrice" + classifieur + ".txt")
	
	return ConfusionMatrice	
	
	
		
def write_in_file(string, fileName):
	file_pointer = open(fileName, "w")
	file_pointer.write(string)
	file_pointer.close()


def examineAllMatrices(allConfusionMatrices, confusionMatriceWithoutACP):
	fig = plt.figure()
	axes = []
	

	nbTotalReconnuSansACP = 0
	nbTotalSansACP = 0
	
	nbTotalReconnuACP = np.zeros(len(allConfusionMatrices))
	nbTotalACP = np.zeros(len(allConfusionMatrices))
	
	x = np.power(range(4, 28+1, 2), 2)
	
	for i in range(0,10):
	
		nbTotalReconnuSansACP += confusionMatriceWithoutACP[i]['nbFound']
		nbTotalSansACP += confusionMatriceWithoutACP[i]['nbTotal']
		
		tauxReconnaissanceSansACP = 100 * confusionMatriceWithoutACP[i]['nbFound']/confusionMatriceWithoutACP[i]['nbTotal']
		ysansACP = np.linspace(tauxReconnaissanceSansACP, tauxReconnaissanceSansACP, len(allConfusionMatrices))
	
		line = int(i/5)+2
		row = i%5
		
		y = []
		
		for j in range(0, len(allConfusionMatrices)):
			tauxReconnaissance = 100 * allConfusionMatrices[j][i]['nbFound']/allConfusionMatrices[j][i]['nbTotal']
			y.append(tauxReconnaissance)
			
			nbTotalReconnuACP[j] += allConfusionMatrices[j][i]['nbFound']
			nbTotalACP[j] += allConfusionMatrices[j][i]['nbTotal']
		
		axes.append(plt.subplot2grid((4, 5), (line, row)))
		axes[i].plot(x, y, '.', x, ysansACP, 'r') 
		axes[i].set_title(convertLbl(i))
		plt.ylim([0, 100])
		
		#on ne voit rien du tout!!!!!!!!!
		'''
		for j in range(0, len(allConfusionMatrices)):
			axes[i].annotate( "{:.2f}".format(y[j]), (x[j],y[j]) , xycoords='data')
		'''
	
	#graphe taux de reconnaissance general
	tauxReconnaissanceSansACP = 100 * nbTotalReconnuSansACP/nbTotalSansACP
	ysansACP = np.linspace(tauxReconnaissanceSansACP, tauxReconnaissanceSansACP, len(allConfusionMatrices))
	
	for j in range(0, len(allConfusionMatrices)):
		y[j] = 100 * nbTotalReconnuACP[j]/nbTotalACP[j]
		
	axeGeneral = plt.subplot2grid((4, 5), (0, 0), colspan=3, rowspan = 2)
	axeGeneral.plot(x, y, '.', x, ysansACP, 'r')
	axeGeneral.set_title("Taux de reconnaissance general")
	plt.ylim([0, 100])
	for j in range(0, len(allConfusionMatrices)):
			axeGeneral.annotate( "{:.2f}".format(y[j]), (x[j],y[j]) , xycoords='data')
	
	
	#graphe taux de reconnaissance general sur temps de calcul
	tauxReconnaissanceSansACP  /= confusionMatriceWithoutACP['TempsCalcul']
	ysansACP = np.linspace(tauxReconnaissanceSansACP, tauxReconnaissanceSansACP, len(allConfusionMatrices))
	
	for j in range(0, len(allConfusionMatrices)):
		y[j] /= allConfusionMatrices[j]['TempsCalcul']
		
	axeGeneral = plt.subplot2grid((4, 5), (1, 3), colspan=2)
	axeGeneral.plot(x, y, '.', x, ysansACP, 'r')
	axeGeneral.set_title("Taux de reconnaissance general sur temps de calcul")
	plt.ylim([0, 100])
	for j in range(0, len(allConfusionMatrices)):
			axeGeneral.annotate( "{:.2f}".format(y[j]), (x[j],y[j]) , xycoords='data')
	
	#graphe temps de calcul
	ysansACP = np.linspace(confusionMatriceWithoutACP['TempsCalcul'], confusionMatriceWithoutACP['TempsCalcul'], len(allConfusionMatrices))
	
	for j in range(0, len(allConfusionMatrices)):
		y[j] = allConfusionMatrices[j]['TempsCalcul']
		
	axeGeneral = plt.subplot2grid((4, 5), (0, 3), colspan=2)
	axeGeneral.plot(x, y, '.', x, ysansACP, 'r')
	axeGeneral.set_title("Temps de calcul")
	for j in range(0, len(allConfusionMatrices)):
			axeGeneral.annotate( "{:.2f}".format(y[j]), (x[j],y[j]) , xycoords='data')
	
	
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
	plt.show()
	return












