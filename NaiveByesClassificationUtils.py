# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:01:34 2020

@author: Santosh Sah
"""

"""
importing the libraries
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importNaiveByesClassificationDataset(naiveByesClassificationDatasetFileName):
    
    naiveByesClassificationDataset = pd.read_csv(naiveByesClassificationDatasetFileName)
    X = naiveByesClassificationDataset.iloc[:, [2, 3]].values
    y = naiveByesClassificationDataset.iloc[:, 4].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveNaiveByesClassificationStandardScaler(naiveByesClassificationStandardScalar):
    
    #Write NaiveByesClassificationStandardScaler in a picke file
    with open("NaiveByesClassificationStandardScaler.pkl",'wb') as NaiveByesClassificationStandardScaler_Pickle:
        pickle.dump(naiveByesClassificationStandardScalar, NaiveByesClassificationStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save NaiveByesClassificationModel as a pickle file.
"""
def saveNaiveByesClassificationModel(naiveByesClassificationModel):
    
    #Write NaiveByesClassificationModel as a picke file
    with open("NaiveByesClassificationModel.pkl",'wb') as NaiveByesClassificationModel_Pickle:
        pickle.dump(naiveByesClassificationModel, NaiveByesClassificationModel_Pickle, protocol = 2)

"""
read NaiveByesClassificationStandardScalar from pickel file
"""
def readNaiveByesClassificationStandardScaler():
    
    #load NaiveByesClassificationStandardScaler object
    with open("NaiveByesClassificationStandardScaler.pkl","rb") as NaiveByesClassificationStandardScaler:
        naiveByesClassificationStandardScalar = pickle.load(NaiveByesClassificationStandardScaler)
    
    return naiveByesClassificationStandardScalar

"""
read NaiveByesClassificationModel from pickle file
"""
def readNaiveByesClassificationModel():
    
    #load NaiveByesClassificationModel model
    with open("NaiveByesClassificationModel.pkl","rb") as NaiveByesClassificationModel:
        naiveByesClassificationModel = pickle.load(NaiveByesClassificationModel)
    
    return naiveByesClassificationModel

"""
read X_train from pickle file
"""
def readNaiveByesClassificationXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readNaiveByesClassificationXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readNaiveByesClassificationYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readNaiveByesClassificationYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveNaiveByesClassificationYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readNaiveByesClassificationYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred