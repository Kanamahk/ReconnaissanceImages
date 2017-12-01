#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt 
from donnees import * 
from sklearn.decomposition import PCA


#if __name__=="__main__":

allConfusionMatrices = []

print("Recupération des données d'entrainement.")
donnees = lectureDonnneesTuple("training")
print("Recupération des données d'entrainement terminé.\nRecupération des données de test.")
donnesAReconnaitre = lectureDonnneesTuple("test")
print("Recupération des données de test terminé.\n")

for i in range(4, 28+1, 2):
	tabImages = getJustImages("training")
	pca = PCA(i*i , True)
	pca.fit(tabImages)

	
	imagesTransforme = pca.transform(donnees[0])
	donneesTransforme = (imagesTransforme, donnees[1])

	dictDonnees = tupleDonnneesToDictList(donneesTransforme)
	averageImagesDict = createAverageImages(dictDonnees)


	imagesTransforme2 = pca.transform(donnesAReconnaitre[0])
	donnesAReconnaitreTransforme = (imagesTransforme2, donnesAReconnaitre[1])
	
	
	allConfusionMatrices.append(makeAllGuess(donnesAReconnaitreTransforme, averageImagesDict, str(i*i)))
	
examineAllMatrices(allConfusionMatrices)

#afiche les averageImage de chaque classe
'''
afficheAverageImages(averageImagesDict, len(averageImagesDict[0]))

'''


