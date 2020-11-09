# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn
#import networkx as nx
import numpy  as np
import pandas as pd
import seaborn as sns; sns.set()
import datetime as dt
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from pandas import compat

compat.PY3 = True
print ("-----------------------------------------------------------------------")
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#load functions from 
from projectFunctions import loadData, missingValues

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Project\Repos\News Popularity\Data'
filename = "OnlineNewsPopularity.csv"
data = loadData(path,filename)

#Check the missing values
misVal, mis_val_table_ren_columns = missingValues(data)
print(mis_val_table_ren_columns.head(20))

#des = data.describe().transpose().to_csv('test.csv')

#Remove rows with missing target values
col = ['url']
data = data.drop(columns=col, axis=1)

from projectFunctions import exploreData, transformData, splitData
exploreData(data)

data_raw = data
features, target = transformData(data_raw)
#des = features.describe().transpose().to_csv('test.csv')

X_train, X_test, y_train, y_test = splitData(features,target,0.3)

from projectFunctions import multinomialnb, svmClassifier, randomForest

#results,learner = multinomialnb(X_train, X_test, y_train, y_test)
#
#print ("Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time']))    
#print ("Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test']))     
#print ("-----------------------------------------------------------------------")
#
#results,learner = svmClassifier(X_train, X_test, y_train, y_test)
#
#print ("Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time']))    
#print ("Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test']))     
#print ("-----------------------------------------------------------------------")

results,learner = randomForest(X_train, X_test, y_train, y_test)
featureImp = pd.DataFrame(columns=['feature','importance'])
featureImp['feature'] = features.columns
featureImp['importance'] = learner.feature_importances_
featureImp = featureImp.sort_values(by='importance')
print(featureImp)
print ("-----------------------------------------------------------------------")
print ("Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time']))    
print ("Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test']))     
print ("-----------------------------------------------------------------------")

#data.to_csv('test.csv',index=False)