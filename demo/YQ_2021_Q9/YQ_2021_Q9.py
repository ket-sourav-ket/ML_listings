import NaiveBayes as nv
import numpy as np
import pandas as pd

df=pd.read_csv('D:\Machine Learning\ML_listings\demo\YQ_2021_Q9\iris.csv',header=None,encoding='utf-8')
Y=df.iloc[0: , 4].values
X=df.iloc[0: , 0:4].values



classifier = nv.NaiveBayesClassifier(X,Y.T)
prediction = classifier.predict(np.array([6.4,3.3,1.9,1.3]))

print('the flower is predicted to be ' + prediction)



