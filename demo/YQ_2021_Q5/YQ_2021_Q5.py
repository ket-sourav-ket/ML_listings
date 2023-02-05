import LinearRegression as lr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def lin_reg_plot(X,Y,model):
    X_stndrd=np.copy(X)
    X_stndrd[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    prediction_std=model.predict(X_stndrd)
    prediction_rescaled=prediction_std*Y[:, 0].std() + Y[:, 0].mean()
    plt.scatter(X,Y,c='blue', edgecolor='white',s=70)
    plt.plot(X,prediction_rescaled,color='black',lw=2)
    return None



df=pd.read_csv('D:\Machine Learning\ML_listings\demo\YQ_2021_Q5\com_sal.csv',header=None,encoding='utf-8')
data=df.iloc[1:,0:].to_numpy(dtype='float32')

X= data[0:,0]
Y= data[0:,1]
Y=np.atleast_2d(Y).T
X=np.atleast_2d(X).T
X_std = np.copy(X)
Y_std = np.copy(Y)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
Y_std[:, 0] = (Y[:, 0] - Y[:, 0].mean()) / Y[:, 0].std()




plt.scatter(X,Y,c='blue', edgecolor='white',s=70)
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show(block=True)

regrObj=lr.LinearRegressionGD()
regrObj.fit(X_std,Y_std)

lin_reg_plot(X,Y,regrObj)
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show(block=True)

std_input= (5- X[:, 0].mean()) / X[:, 0].std()
std_ideal_salary=regrObj.predict(np.transpose(np.atleast_2d(np.array([std_input]))))
ideal_salary=(std_ideal_salary*Y[:, 0].std()) + Y[:, 0].mean()
print(f"ideal salary is {ideal_salary}")