import LinearRegression as lr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lin_reg_plot(X,Y,model):
    plt.scatter(X,Y,c='blue', edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None



df=pd.read_csv('D:\Machine Learning\ML_listings\demo\YQ_2021_Q5\com_sal.csv',header=None,encoding='utf-8')
data=df.iloc[1:,0:].to_numpy(dtype='float32')

X= data[0:,0]
Y= data[0:,1]


plt.scatter(X,Y,c='blue', edgecolor='white',s=70)
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show(block=True)

Y=np.atleast_2d(Y)
X=np.atleast_2d(X)

print(X.T)

regrObj=lr.LinearRegressionGD()
regrObj.fit(X.T,Y.T)

lin_reg_plot(X.T,Y,regrObj)
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show(block=True)

ideal_salary=regrObj.predict(np.transpose(np.atleast_2d(np.array([5]))))
print(f"ideal salary is {ideal_salary}")







