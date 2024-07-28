import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import os

#784 x 10 x 10 Neural Network for MNIST, digit recognizer

def shuffle_with_seed(arr, seed):
    #shuffle the rows of the matrix
    np.random.seed(seed)
    np.random.shuffle(arr)
    return arr


data = pd.read_csv('train.csv')
data = np.array(data)
M,N = data.shape
data = shuffle_with_seed(data,42)

#manual train-test split
data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:N]
X_test = X_test/255
#normalize the X value so that it deos not blow up in the exponential calculations.

data_train = data[1000:M].T
Y_train = data_train[0]
X_train = data_train[1:N]
X_train = X_train/255

N_train,M_train = X_train.shape
N0 = N_train
N1 = 10
N2 = 10
M = M_train


def initialize_para(seed=None):
    # pass the random seed for reproducibility
    W1 = generate_random_matrix(N1,N0,seed)
    W2 = generate_random_matrix(N2,N1,seed)
    b1 = generate_random_matrix(N1,1,seed)
    b2 = generate_random_matrix(N2,1,seed)

    W = (W1,W2)
    b = (b1,b2)
    
    return W, b


def generate_random_matrix(N, M, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    # Generate an N x M matrix with values uniformly distributed between 0 and 1
    random_matrix = np.random.rand(N, M)
    # Scale and shift the values to the range [-0.5, 0.5]
    random_matrix = 1 * random_matrix - 0.5
    return random_matrix

def forward_propagation(W,b,X):
    b1, b2 = b
    W1, W2 = W
    Z1 = W1.dot(X)+b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softmax(Z2)

    Z = (Z1,Z2)
    A = (A1,A2)

    return Z,A


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Subtract max for numerical stability
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def ReLU(Z):
    return np.maximum(0,Z)


def ReLU_derivative(Z):
    return Z>0

#pack A1,A2 into tuples(can also use lists)
#A = (A1,A2)

#calculation of gradient terms
def back_propagation(A,Z,W,X,Y):
    #A0 = X #size N0 x M
    A1, A2 = A
    Z1, Z2 = Z
    W1, W2 = W
    #N,M = X.shape
    # apply one hot encoding to Y
    one_hot_Y = one_hot(Y)
    #calculate the approximate form for partial L partial Z2
    pd_Z2_L = (A2-one_hot_Y)/M #size N2 x M
    #calculate the approximate form for partial L partial Z1
    pd_Z1_L = W2.T.dot(pd_Z2_L)*ReLU_derivative(Z1) # size N1 x M

    # now the derivative that directly affect gradient updates
    pd_W2_L = pd_Z2_L.dot(A1.T) #size N2 x M
    pd_b2_L = np.sum(pd_Z2_L, axis = 1, keepdims=True) #size N2 x 1

    pd_W1_L = pd_Z1_L.dot(X.T) #size N1 x M
    pd_b1_L = np.sum(pd_Z1_L, axis=1, keepdims=True) # size N1 x 1

    pd_W = (pd_W1_L, pd_W2_L)
    pd_b = (pd_b1_L, pd_b2_L)
    return pd_W, pd_b

def update_weights(W, b, pd_W, pd_b, alpha):
    W = tuple(w - alpha * pd_w for w, pd_w in zip(W, pd_W))
    b = tuple(bi - alpha * pd_bi for bi, pd_bi in zip(b, pd_b))
    return W, b

#one hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def gradient_descent(X,Y,alpha,iter,seed=None,W=None,b=None):
    if W is None and b is None:
        W, b = initialize_para(seed)

    i=0
    while i < iter:
        Z,A = forward_propagation(W,b,X)
        pd_W, pd_b = back_propagation(A,Z,W,X,Y)
        W,b = update_weights(W,b,pd_W,pd_b,alpha)
        i = i+1
        if (i%10 ==0 and i<=50) or i%100 ==0:
            print("iteration: %d\n" %i)
            pred_Y = get_pred(A)
            accuracy = get_accuracy(Y,pred_Y)
            print('%s || %s\naccuracy = %.6f' % (pred_Y, Y, accuracy))
    return W,b

def get_pred(A):
    _,A2 = A
    # Use np.argmax along axis 0 to get the indices of the maximum values in each column
    max_indices = np.argmax(A2, axis=0)
    return max_indices

def get_accuracy(Y,pred_Y):
    return np.sum(Y==pred_Y, axis=0)/len(Y)


def get_new_pred(X,W,b):
    _,A = forward_propagation(W,b,X)
    Y_pred = get_pred(A)
    return Y_pred

def show_pred(idx_sample,X,Y, W,b):
    current_sample_X = X[:, idx_sample,None] # the None keeps the structure of the slicing, i.e. keeps the result as a column vector
    current_sample_Y = Y[idx_sample]
    current_pred_Y = get_new_pred(current_sample_X,W,b)
    print('prediction:',current_pred_Y)
    print('lable:',current_sample_Y)

    X_mat = current_sample_X.reshape((28, 28)) * 255
    #plt.figure(figsize=(5, 5))
    plt.imshow(X_mat, cmap='gray')
    plt.show()



#W,b = gradient_descent(X_train,Y_train,0.2,1000,42,)

seed_num = 42

filename = 'weights_biases0.pkl'
# Check if the file exists
if os.path.exists(filename):
    print(f"The file '{filename}' exists. Using existing W and b for continued training.")
    with open(filename, 'rb') as file:
        W, b = pickle.load(file)
    W,b = gradient_descent(X_train,Y_train,0.1,1000,seed_num,W,b)
else:
    print(f"The file '{filename}' does not exist. Star training from randomly initialized parameters.")
    W,b = gradient_descent(X_train,Y_train,0.1,1000,seed_num,None,None)

print('Iterations complete, saving training progress.')
with open(filename, 'wb') as file:
    pickle.dump((W, b), file)
print(f"Lists saved to '{filename}'.")