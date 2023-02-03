import LinearRegression as lr
import numpy as np
import matplotlib.pyplot as plt
def fn_y(x):
    return np.cos(x) + np.sin(x)

def plot_fn():
    X=np.arange(0,2*np.pi,0.1)
    Y=fn_y(X)
    plt.plot(X,Y)
    plt.ylabel("sin(x) + cos(x)")
    plt.xlabel("x")
    plt.show(block=True)

def lin_reg_plot(X,Y,model):
    plt.scatter(X,Y,c='blue', edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None



plot_fn()


X=np.transpose(np.atleast_2d(np.array([0,1,2,3,4])))
Y=fn_y(X)

LiReObj = lr.LinearRegressionGD()
LiReObj.fit(X,Y)



lin_reg_plot(X,Y.T,LiReObj)
plt.xlabel('x:independent variable')
plt.ylabel('fn(x)')
plt.show(block=True)

X_test=np.transpose(np.atleast_2d(np.array([5,6,7,8,9])))
Y_pred = LiReObj.predict(X_test)
print(Y_pred)
plt.scatter(X_test,Y_pred,c='blue', edgecolor='white',s=70)
plt.show(block=True)
