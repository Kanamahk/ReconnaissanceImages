#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt 

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
	res = np.zeros(784)
	for i in range(0, 784):
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
	for i in range(0, 784):
		temp = averageImage[i] - image[i]
		res += -temp if temp < 0 else temp
	return res

def calcAllDifference(image, averageImages):
	total = 0
	tailleDuPlusPetit = 255*784 #difference max
	plusPetit= 0;
	Dict = {}
	for i in range(0, len(averageImages)):
		Dict[i] = calcDifference(image, averageImages[i])
		if Dict[i] < tailleDuPlusPetit :
			tailleDuPlusPetit = Dict[i]
			plusPetit = i
		total += Dict[i]
	return (plusPetit, Dict, total)

def makeAGuess(data, averageImages):
	difs = calcAllDifference(data[0], averageImages)
	#proba = 100 * (1- difs[1][difs[0]]/difs[2])
	#print('Je pense que ce vetement est un ' + convertLbl(difs[0]) + ' avec une probabilite de ' + str(proba) + '%.')
	print('Je pense que ce vetement est un ' + convertLbl(difs[0]) + '.')
	if difs[0] == data[1]:
<<<<<<< HEAD
		#print('Ce vetement est effectivement un ' + convertLbl(difs[0]) + '.')
		string = string + 'Ce vetement est effectivement un ' + convertLbl(difs[0]) + '.\n\n'
		
		Dict[difs[0]]['nbFound'] += 1
		res = 1
	else:
		#print('Ce vetement etait en fait un ' + convertLbl(data[1]) + '.')
		string = string + 'Ce vetement etait en fait un ' + convertLbl(data[1]) + '.\n\n'
=======
		print('Ce vetement est effectivement un ' + convertLbl(difs[0]) + '.')
		res = 1
	else:
		print('Ce vetement etait en fait un ' + convertLbl(data[1]) + '.')
>>>>>>> origin/master
		res = 0
	print()
	return res

def makeAllGuess(datas, averageImages, name):
	nbReconnu = 0
	for i in range(0, len(datas)):
		nbReconnu += makeAGuess(datas[i], averageImages)
	print('\nJ\'ai reconnu ' + str(nbReconnu) + ' vetements sur ' + str(len(datas)) + ' (' + str(100 * nbReconnu/len(datas)) + '%).')
		
<<<<<<< HEAD

	#print('\nJ\'ai reconnu ' + str(nbReconnu) + ' vetements sur ' + str(len(datas)) + ' (' + str(100 * nbReconnu/len(datas)) + '%).')
	string = string + '\nJ\'ai reconnu ' + str(nbReconnu) + ' vetements sur ' + str(len(datas[0])) + ' (' + str(100 * nbReconnu/len(datas[0])) + '%).'
	write_in_file(string, "res" + name +".txt")
	return CofusionMatrice

		
def write_in_file(string, fileName):
	file_pointer = open(fileName, "w")
	file_pointer.write(string)
	file_pointer.close()
=======
		
	
	



>>>>>>> origin/master


def examineAllMatrices(allConfusionMatrices):
	fig = plt.figure()
	axes = []
	
	x = np.power(range(4, 28+1, 2), 2)

	nbTotalReconnu = np.zeros(len(allConfusionMatrices))
	nbTotal = np.zeros(len(allConfusionMatrices))
	
	for i in range(0,10):
		line = int(i/5)+2
		row = i%5
		
		y = []
		
		for j in range(0, len(allConfusionMatrices)):
			tauxReconnaissance = 100 * allConfusionMatrices[j][i]['nbFound']/allConfusionMatrices[j][i]['nbTotal']
			y.append(tauxReconnaissance)
			
			nbTotalReconnu[j] += allConfusionMatrices[j][i]['nbFound']
			nbTotal[j] += allConfusionMatrices[j][i]['nbTotal']
		
		axes.append(plt.subplot2grid((4, 5), (line, row)))
		axes[i].plot(x, y, '.') 
		axes[i].set_title(convertLbl(i))
		plt.ylim([0, 100])
		
		#on ne voit rien du tout!!!!!!!!!
		'''
		for j in range(0, len(allConfusionMatrices)):
			axes[i].annotate( "{:.2f}".format(y[j]), (x[j],y[j]) , xycoords='data')
		'''
	
	for j in range(0, len(allConfusionMatrices)):
		y[j] = 100 * nbTotalReconnu[j]/nbTotal[j]
		
	axeGeneral = plt.subplot2grid((4, 5), (0, 0), colspan=5, rowspan = 2)
	axeGeneral.plot(x, y, '.')
	axeGeneral.set_title("Scores total")
	plt.ylim([0, 100])
	for j in range(0, len(allConfusionMatrices)):
			axeGeneral.annotate( "{:.2f}".format(y[j]), (x[j],y[j]) , xycoords='data')
	
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
	plt.show()
	return












