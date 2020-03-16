# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:27:01 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from NaiveByesClassificationUtils import (readNaiveByesClassificationYTest, readNaiveByesClassificationYPred)

"""

calculating NaiveByesClassification regression confussion matrix

"""
def testNaiveByesClassificationConfussionMatrix():
    
    y_test = readNaiveByesClassificationYTest()
    y_pred = readNaiveByesClassificationYPred()
    
    naiveByesClassificationConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(naiveByesClassificationConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[55  3]
    [ 4 18]]
    
    """
"""
calculating accuracy score

"""

def testNaiveByesClassificationAccuracy():
    
    y_test = readNaiveByesClassificationYTest()
    y_pred = readNaiveByesClassificationYPred()
    
    naiveByesClassificationConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(naiveByesClassificationConfussionAccuracy) #.9125%

"""
calculating classification report

"""

def testNaiveByesClassificationClassificationReport():
    
    y_test = readNaiveByesClassificationYTest()
    y_pred = readNaiveByesClassificationYPred()
    
    naiveByesClassificationConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(naiveByesClassificationConfussionClassificationReport)
    
    """
             precision    recall  f1-score   support

          0       0.93      0.95      0.94        58
          1       0.86      0.82      0.84        22

avg / total       0.91      0.91      0.91        80
    """
    
if __name__ == "__main__":
    #testNaiveByesClassificationConfussionMatrix()
    #testNaiveByesClassificationAccuracy()
    testNaiveByesClassificationClassificationReport()