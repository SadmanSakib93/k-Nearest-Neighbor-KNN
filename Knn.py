# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:17:39 2019

@author: Sadman Sakib
"""
import random
from os import walk
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
from numpy import linalg as LA
import operator

path_training='dataset1\\training_validation'
path_testing='dataset1\\test'
traingClassLabels=[]

# Reads the name of all the files in the path
def readFileNames(path):
    for dirpath, dirnames, filenames in walk(path):
        print("")
#    print(filenames)
    return filenames

# Reads the content of the specified file
def readFileContent(path, fileName):
    f = open(str(path)+"\\"+str(fileName), "r")
    fileContent=f.read()
    return fileContent

# Prepare dataset instances and class labels from the files
def prepareTrainingArr(path,files):
    trainDataset=[]
    random.shuffle(files)
#    print((files))
    for eachFile in files:
        row=[]
        fileContent=(readFileContent(path,eachFile))
        fileName=eachFile
# Find the class label from the file name
        classLabel=fileName[6:7]
        fileContent = fileContent.replace('\r', '').replace('\n', '')
#        traingClassLabels.append(classLabel)
#        print(len(fileContent))
        for bit in (fileContent):
            row.append(int(bit))
        trainDataset.append(row)
    return trainDataset
    
def distanceMeasure(row1,row2):
    a = np.asarray(row1) 
    b = np.asarray(row2) 
    return (LA.norm(a-b))

# 5-Fold Cross-Validation 
# K varied from 1 to 11
kAccAverage=[] 
def findBestk(files,trainDataset):
    kAccFor5Folds=[]
    for k in range(1,12):
        fold=1
        row=[]
        kf = KFold(n_splits=5,shuffle=False)
        for train_index, test_index in kf.split(files):
#            print('Fold:'+str(fold))
            accSumforEachFold=0    
            for testIndx in test_index:
                distances = []
                kthNeighbor=[]
                correctCount=0
                for trainIndx in train_index:
                    distances.append((files[testIndx],files[trainIndx],distanceMeasure(trainDataset[testIndx],trainDataset[trainIndx])))
                distances.sort(key = operator.itemgetter(2))    
                for j in range(k):
                    kthNeighbor.append(distances[j])
                for nearests in kthNeighbor:
                    actualName=nearests[0]
                    predictedName=nearests[1]
# Checking if the prediction is correct or not
                    if(actualName[6:7]==predictedName[6:7]):
                        correctCount=correctCount+1
                accSumforEachFold=accSumforEachFold+(correctCount/k)
#            print("K = "+str(k)+"| ACC for fold "+str(fold)+" = "+str(accSumforEachFold/len(test_index)))
            row.append(accSumforEachFold/len(test_index))
            fold=fold+1
        kAccFor5Folds.append(row)   
    print("K\tResults") 
    for row in range(len(kAccFor5Folds)):
        sum=0
        for val in ((kAccFor5Folds[row])):
            sum=sum+val
        avg=sum/5
        print(str(row+1)+"\t"+str(avg))
        kAccAverage.append(avg)        
    best_K=kAccAverage.index(max(kAccAverage))
    print("Selected best K value:"+str(best_K+1))   
    return (best_K+1)

def calculateAccForAllK(files, trainDataset, testFiles, testDataset):
    best_K=findBestk(files,trainDataset)
    print("K\tResults") 
    for k in range(1,12):
        accuracyForEachTestData=[]
        actualClass=[]
        predictedClass=[]
        for test_indx in range(len(testDataset)):
            distance=[]
            kthNeighbor=[]
            correctCount=0
            for train_indx in range(len(trainDataset)):
                distance.append((testFiles[test_indx],files[train_indx],distanceMeasure(testDataset[test_indx],trainDataset[train_indx])))
            distance.sort(key = operator.itemgetter(2))   
            for j in range(k):
                kthNeighbor.append(distance[j])    
            for nearests in kthNeighbor:
                actualName=nearests[0]
                predictedName=nearests[1]
                actualClass.append(actualName[6:7])
                predictedClass.append(predictedName[6:7])
                if(actualName[6:7]==predictedName[6:7]):
                    correctCount=correctCount+1
            accuracyForEachTestData.append(correctCount/(k))
        #    print("***TEST RESULTS "+testFiles[test_indx]+" ***")
        averageAccuracy=(sum(accuracyForEachTestData)/len(accuracyForEachTestData))
        print(str(k)+"\t"+str(averageAccuracy))
    #    print("FINAL ACCURACY for K value "+str(k)+"="+str(averageAccuracy))
    #    print("Confusion Matrix:")
    #    print(confusion_matrix(actualClass, predictedClass))               

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

files=readFileNames(path_training)
trainDataset=(prepareTrainingArr(path_training,files))
trainDataset = min_max_scaler.fit_transform(trainDataset)

testFiles=readFileNames(path_testing)
testDataset=(prepareTrainingArr(path_testing,testFiles))
testDataset = min_max_scaler.fit_transform(testDataset)
calculateAccForAllK(files, trainDataset, testFiles, testDataset)
     
        
        
        
        
        