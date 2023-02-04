import numpy as np
import matplotlib.pyplot as plt
import sys

X=np.array([[1,2,3,4],[3,4,8,9],[1,3,5,4],[6,3,8,1]])

#if library method is allowed do this:
inv_x=np.linalg.inv(X)


#else if library method is not allowed do all of these:

def reshape(X,row_pos,col_pos):
    arr=np.delete(X,row_pos,0)
    arr=np.delete(arr,col_pos,1)
    return arr

def minor(X,row_pos,col_pos):
    sub_arr=reshape(X,row_pos,col_pos)
    return det(sub_arr)


def det(X):
    if(X.shape[0]==2):
        return ((X[0,0]*X[1,1])-(X[0,1]*X[1,0]))

    else:
        det_val=0
        for col_i in range(X.shape[0]):
            det_val = det_val + (-1**col_i)*X[0,col_i]*det(reshape(X,0,col_i))
        return det_val

def create_mask(shape_tuple):
    cf_mask=np.ones(shape_tuple)
    for i in range(shape_tuple[0]):
        for j in range(shape_tuple[0]):
            if(((i+j)%2)!=0):
                cf_mask[i,j]=-1
    
    return cf_mask

def adjoint(X):
    mat_of_minors= np.ones(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            mat_of_minors[i][j]=minor(X.T,i,j)
    
    return(np.multiply(mat_of_minors,create_mask(X.shape)))
def inverse(X):
    det_X=det(X)
    adj_X=adjoint(X)
    try:
        inv_X=(1/det_X)*adj_X
        return inv_X
    except ZeroDivisionError:
        print("singular matrix")
        sys.exit()

print(inverse(X))

sum_x= X + inverse(X)
pro_x= X * inverse(X)
dif_x = X - inverse(X)