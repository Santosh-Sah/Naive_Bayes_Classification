# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:15:53 2020

@author: Santosh Sah
"""

import pandas as pd
from NaiveByesClassificationUtils import readNaiveByesClassificationModel, readNaiveByesClassificationStandardScaler

def predict():
    
    naiveByesClassification = readNaiveByesClassificationModel()
    naiveByesClassificationStandardScaler = readNaiveByesClassificationStandardScaler()
    
    inputValue = [[26, 1000]]
    inputValueDataframe = pd.DataFrame(naiveByesClassificationStandardScaler.transform(inputValue))
    
    predictedValue = naiveByesClassification.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()