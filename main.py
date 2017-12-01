#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt 
from donnees import * 
from sklearn.decomposition import PCA
import timeit


#if __name__=="__main__":

allConfusionMatrices = []

print("Recupération des données d'entrainement.")
donnees = lectureDonnneesTuple("training")
print("Recupération des données d'entrainement terminé.\nRecupération des données d'evaluation.")
donnesAReconnaitre = lectureDonnneesTuple("test")
print("Recupération des données d'evaluation terminé.\n")

tabImages = getJustImages("training")

for i in range(4, 28+1, 2):
	start_time = timeit.default_timer()
	print("*************************************************************************")
	print("Creation de la matrice de reduction à partir des donnees d'entrainement : parametre de dimension est :" + str(i*i) + ".")
	pca = PCA(i*i , True)
	pca.fit(tabImages)
	print("Matrice créée.\n")
	
	print("Transformation des données d'entrainement.")
	imagesTransforme = pca.transform(donnees[0])
	donneesTransforme = (imagesTransforme, donnees[1])
	print("Transformation des données d'entrainement terminé.\n")

	print("Création des images moyennes de chaque groupe.")
	dictDonnees = tupleDonnneesToDictList(donneesTransforme)
	averageImagesDict = createAverageImages(dictDonnees)
	print("Création des images moyennes de chaque groupe terminé.\n")

	print("Transformation des données d'evaluation.")
	imagesTransforme2 = pca.transform(donnesAReconnaitre[0])
	donnesAReconnaitreTransforme = (imagesTransforme2, donnesAReconnaitre[1])
	print("Transformation des données d'evaluation terminé.\n")
	
	tempsPreparation = timeit.default_timer() - start_time
	
	print("Commencement de la reconnaissance.")
	allConfusionMatrices.append(makeAllGuess(donnesAReconnaitreTransforme, averageImagesDict, str(i*i), tempsPreparation))
	print("Reconnaissance terminé. Deux fichiers contenant les résultats et la matrice de confusion ont été créé.\n")

print("*************************************************************************")
print("Toutes les reconnaissances sont terminé : affichage des données sous forme de graphe.")
	
examineAllMatrices(allConfusionMatrices)

#afiche les averageImage de chaque classe
'''
afficheAverageImages(averageImagesDict, len(averageImagesDict[0]))

'''



