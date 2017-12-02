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


'''
print("*************************************************************************")
print("Classifieur à distance minimum sans ACP.")
start_time = timeit.default_timer()
print("Création des images moyennes de chaque groupe.")
dictDonnees = tupleDonnneesToDictList(donnees)
averageImagesDict = createAverageImages(dictDonnees)
print("Création des images moyennes de chaque groupe terminé.\n")
tempsPreparation = timeit.default_timer() - start_time


print("Commencement de la reconnaissance.")
confusionMatriceWithoutACP = makeAllGuess(donnesAReconnaitre, averageImagesDict, 'SansACP', tempsPreparation)
print("Reconnaissance terminé. Deux fichiers contenant les résultats et la matrice de confusion ont été créé.\n")



tabImages = getJustImages("training")

for i in range(4, 28+1, 2):
	start_time = timeit.default_timer()
	print("*************************************************************************")
	print("Classifieur à distance minimum avec ACP avec comme parametre de dimension " + str(i*i) + ".")
	print("Creation de la matrice de reduction à partir des donnees d'entrainement.")
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
	allConfusionMatrices.append(makeAllGuessDMIN(donnesAReconnaitreTransforme, averageImagesDict, 'ACP' + str(i*i), tempsPreparation))
	print("Reconnaissance terminé. Deux fichiers contenant les résultats et la matrice de confusion ont été créé.\n")
'''

'''
print("*************************************************************************")
print("Classifieur LinearSVM.")
print("Commencement de l'entrainement.")
start_time = timeit.default_timer()
lin_clf = svm.LinearSVC()
lin_clf.fit(donnees[0], donnees[1])
tempsPreparation = timeit.default_timer() - start_time
print("Entrainement terminé.")
confusionMatriceLSVM = makeAllGuessLSVM_NNC(lin_clf, donnesAReconnaitre, tempsPreparation, "LinearSVM")
'''

print("*************************************************************************")
print("Classifieur plus proche voisin.")
print("Commencement de l'entrainement.")
start_time = timeit.default_timer()
clf =  neighbors.KNeighborsClassifier()
clf.fit(donnees[0], donnees[1])
tempsPreparation = timeit.default_timer() - start_time
print("Entrainement terminé.")
confusionMatriceNNC = makeAllGuessLSVM_NNC(clf, donnesAReconnaitre, tempsPreparation, "NNC")

'''
print("*************************************************************************")
print("Toutes les reconnaissances sont terminé : affichage des données sous forme de graphe.")
	
examineAllMatrices(allConfusionMatrices, confusionMatriceWithoutACP)
'''
#afiche les averageImage de chaque classe
'''
afficheAverageImages(averageImagesDict, len(averageImagesDict[0]))

'''



