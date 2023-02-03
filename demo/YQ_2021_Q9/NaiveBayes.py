"""
@uthor:sourav

Naive Bayes classifier for  linearly seperable categorical data

Note: The various likelihood estimates(class conditional probabilities) are
computed directly after test data(or new data) is provided.

"""
import numpy as np

class NaiveBayesClassifier(object):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y

    def classProb(self,Y):      #general class probabilities(prior) estimate
        self.classL_ = np.unique(Y)
        self.classProb_ = np.array([])
        for i in range(self.classL_.shape[0]):
            tempLabelcount=0
            for j in range(Y.shape[0]):
                if self.classL_[i]==Y[j]:
                    tempLabelcount +=1
            
            self.classProb_=np.append(self.classProb_,[tempLabelcount/(Y.shape[0])])

    def likelihoodEs(self,featureIndx,featureVal,classLabel):
        tempClassCount=0
        tempFeatureCount=0

        for i in range(self.Y.shape[0]):
            if (classLabel==self.Y[i]):
                tempClassCount += 1
                if(featureVal == self.X[i,featureIndx]):
                    tempFeatureCount +=1

        return (tempFeatureCount/tempClassCount)   #maximum likelihood estimate
    
    def predict(self,X):
        classCondProb = np.array([])
        self.classProb(self.Y)
        for i in range(self.classL_.shape[0]):
            tempClassCondProb=1
            for j in range(X.shape[0]):
                tempClassCondProb*= self.likelihoodEs(j,X[j],self.classL_[i])
            
            classCondProb=np.append(classCondProb,[tempClassCondProb])  #likelihood of the featue vector given the various class labels
        
        
        posteriorProb=self.classProb_ * classCondProb

        return (self.classL_[np.argmax(posteriorProb)])




                


            


