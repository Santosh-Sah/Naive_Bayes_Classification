# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:47:27 2020

@author: Santosh Sah
"""

from NaiveByesClassificationUtils import (readNaiveByesClassificationXTest, readNaiveByesClassificationModel,
                                     saveNaiveByesClassificationYPred, readNaiveByesClassificationStandardScaler)

"""
test the model on testing dataset
"""
def testNaiveByesClassificationModel():
    
    X_test = readNaiveByesClassificationXTest()
    naiveByesClassificationStandardScaler = readNaiveByesClassificationStandardScaler()
    X_test = naiveByesClassificationStandardScaler.transform(X_test)
    
    naiveByesClassificationModel = readNaiveByesClassificationModel()
    
    y_pred = naiveByesClassificationModel.predict(X_test)
    saveNaiveByesClassificationYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testNaiveByesClassificationModel()