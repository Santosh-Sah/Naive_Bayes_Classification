# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:17:32 2020

@author: Santosh Sah
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from NaiveByesClassificationUtils import (saveNaiveByesClassificationModel, readNaiveByesClassificationXTrain, readNaiveByesClassificationYTrain,
                                     saveNaiveByesClassificationStandardScaler)

"""
Train NaiveByesClassification model 
"""
def trainNaiveByesClassificationModel():
    
    naiveByesClassificationStandardScalar = StandardScaler()
    
    X_train = readNaiveByesClassificationXTrain()
    y_train = readNaiveByesClassificationYTrain()
    
    naiveByesClassificationStandardScalar.fit(X_train)
    saveNaiveByesClassificationStandardScaler(naiveByesClassificationStandardScalar)
    
    X_train = naiveByesClassificationStandardScalar.transform(X_train)
    
    naiveByesClassification = GaussianNB()
    naiveByesClassification.fit(X_train, y_train)
    
    saveNaiveByesClassificationModel(naiveByesClassification)

if __name__ == "__main__":
    trainNaiveByesClassificationModel()