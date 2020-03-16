# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:22:06 2020

@author: Santosh Sah
"""

from NaiveByesClassificationUtils import (importNaiveByesClassificationDataset, saveTrainingAndTestingDataset)

def preprocess():
    
    X_train, X_test, y_train, y_test = importNaiveByesClassificationDataset("Naive_Byes_Classification_Social_Network_Ads.csv")
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()