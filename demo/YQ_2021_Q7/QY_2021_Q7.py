import pandas as pd
import numpy as np
import NeuralNet_MLP as nn


df=pd.read_csv('D:\Machine Learning\ML_listings\demo\YQ_2021_Q7\iris.csv',header=None,encoding='utf-8')
Y=df.iloc[0: , 4].values
Y[0:50]=0
Y[50:100]=1
Y[100:150]=2
X=df.iloc[0: , 0:4].values
class_labels=['iris-setosa','iris-versicolor','iris-virginica']
network = nn.NeuralNetMLP(n_hidden=60,l2=0.01,epochs=100,eta=0.005,seed=1)
network.fit(X[0:148],Y[0:148].T)
prediction = network.predict(X[148:])

for i in range(2):
    print(f"the feature vector {X[148+i]} is predicted to be {class_labels[prediction[i]]}")