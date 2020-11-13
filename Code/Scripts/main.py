# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn
import pandas as pd
import seaborn as sns; sns.set()
from pandas import compat

compat.PY3 = True
print ("-----------------------------------------------------------------------")
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#load functions from 
from projectFunctions import loadData, missingValues

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Project\Repos\News Popularity\Git\EECS-731-Project-7\Data'
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

from projectFunctions import multinomialnb, svmClassifier, randomForest, pca, gclus

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

#results,learner = randomForest(X_train, X_test, y_train, y_test)
#featureImp = pd.DataFrame(columns=['feature','importance'])
#featureImp['feature'] = features.columns
#featureImp['importance'] = learner.feature_importances_
#featureImp = featureImp.sort_values(by='importance')
#print(featureImp)
#print ("-----------------------------------------------------------------------")
#print ("Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time']))    
#print ("Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test']))     
#print ("-----------------------------------------------------------------------")

dim = 2
reduced_f, pca_comp, pca = pca(features,dim)
from pics import pcadim, biplot
#pcad = pca_comp['D0']
#pcad = pca_comp.iloc[:,0]
#pcadim(pcad)
# Create a DataFrame for the reduced data
reduced_f = pd.DataFrame(reduced_f, columns = ['Dimension 1', 'Dimension 2'])
biplot(features, reduced_f, pca)
#clus_df = pd.DataFrame(columns=['clusters','score'])
#clist = [30]; slist = [];
#for dim in clist:
#    cluster, centers, score = gclus(reduced_f,dim)
#    slist.append(score * 100)
#    print ("-----------------------------------------------------------------------")
#    print "silhouette score for GMM: {:.4f}".format(score)
#    print "Optimal number of components: {:.4f}".format(cluster.n_components)
#    print "number of centers: {:.4f}".format(len(centers))
#    print ("-----------------------------------------------------------------------")
#
#clus_df['clusters'] = clist
#clus_df['score'] = slist
#
#ax = sns.barplot(x="clusters", y="score", data=clus_df)
#ax.set_title('Clustering scores')

#data.to_csv('test.csv',index=False)