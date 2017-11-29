#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt 
from donnees import * 
from sklearn.decomposition import PCA


#if __name__=="__main__":


tabImages = getJustImages("training")
pca = PCA(600, True)
pca.fit(tabImages)


donnees = lectureDonnneesTuple("training")
imagesTransforme = pca.transform(donnees[0])
donnees = (imagesTransforme, donnees[1])

dictDonnees = tupleDonnneesToDictList(donnees)
averageImagesDict = createAverageImages(dictDonnees)


donnesAReconnaitre = lectureDonnneesTuple("test")
imagesTransforme2 = pca.transform(donnesAReconnaitre[0])
donnesAReconnaitre = (imagesTransforme2, donnesAReconnaitre[1])

makeAllGuess(donnesAReconnaitre, averageImagesDict)


#afiche les averageImage de chaque classe
'''
afficheAverageImages(averageImagesDict)

'''


