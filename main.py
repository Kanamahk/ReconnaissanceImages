#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt 
from donnees import * 


#if __name__=="__main__":


donnees = lectureDonnneesTuple("dev")
dictDonnees = tupleDonnneesToDictList(donnees)
averageImagesDict = createAverageImages(dictDonnees)

donnesAReconnaitre = lectureDonnneesList("test")

makeAllGuess(donnesAReconnaitre, averageImagesDict)

'''
#afiche les averageImage de chaque classe
afficheAverageImages(averageImagesDict)
'''



