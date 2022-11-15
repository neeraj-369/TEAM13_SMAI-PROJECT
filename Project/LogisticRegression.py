import numpy as np
from sklearn.linear_model import LogisticRegressionCV


def Logistic_Regression():
    Cs = [0.001,0.01,0.1,1,10,100,1000]
    logistic_classifier = LogisticRegressionCV(Cs=Cs,penalty='l2',solver='sag',cv=5)
    