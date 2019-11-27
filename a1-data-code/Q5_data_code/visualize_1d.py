#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:15]
#x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

#feature=np.matrix('1,2,3;4,5,6')


degree=3

train_err = dict()
test_err = dict()

def func_x_times(martrix,deg):
    x_zero_time=np.ones(martrix.shape[0],dtype=int)

    x_zero_transpose=x_zero_time.reshape(x_zero_time.shape[0],1)

    new_martrix=martrix
    martrix = np.concatenate((x_zero_transpose, martrix), 1)
    if deg==1:
        return martrix
    else:
        for n in range(2,deg+1):
            x_times = np.power(new_martrix, n)
            martrix=np.concatenate((martrix,x_times),1)

        return martrix
def no_bias(martrix,deg):
    new_martrix=martrix

    if deg == 1:
        return martrix
    else:

        for n in range(2, deg+1):
            x_times = np.power(new_martrix,n)
            martrix = np.concatenate((martrix, x_times), 1)

        return martrix
def plot_feature(feature):
    x_train = x[0:N_TRAIN,feature-7]
    x_test = x[N_TRAIN:,feature-7]
    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    train_bigfai = func_x_times(x_train, 3)
    w = np.linalg.pinv(train_bigfai) * t_train
    bigfai_ev=func_x_times(np.transpose(np.asmatrix(x_ev)),3)
    y_ev= np.transpose(w) * np.transpose(bigfai_ev)
    plt.plot(x_train,t_train,'bo')
    plt.plot(x_test,t_test,'go')
    plt.plot(x_ev,np.transpose(y_ev),'r.-')
    plt.legend(['Training data','Test data','Learned Polynomial'])
    plt.title('A visualization of a regression estimate using random outputs')
    plt.show()
def plot_feature_nobias(feature):
    x_train = x[0:N_TRAIN,feature-7]
    x_test = x[N_TRAIN:,feature-7]
    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    train_bigfai = no_bias(x_train, 3)
    w = np.linalg.pinv(train_bigfai) * t_train
    bigfai_ev=no_bias(np.transpose(np.asmatrix(x_ev)),3)
    y_ev= np.transpose(w) * np.transpose(bigfai_ev)
    plt.plot(x_train,t_train,'bo')
    plt.plot(x_test,t_test,'go')
    plt.plot(x_ev,np.transpose(y_ev),'r.-')
    plt.title('A visualization of a regression estimate using random outputs')
    plt.show()
plot_feature(12)

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
#x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
#x_ev = np.linspace(np.asscalar(min(min(x_train[:,f]),min(x_test[:,f]))),
#                   np.asscalar(max(max(x_train[:,f]),max(x_test[:,f]))), num=500)
'''x1_ev = np.linspace(0, 10, num=500)
x2_ev = np.linspace(0, 10, num=50)'''

# TO DO::
# Perform regression on the linspace samples.
# Put your regression estimate here in place of y_ev.
'''y1_ev = np.random.random_sample(x1_ev.shape)
y2_ev = np.random.random_sample(x2_ev.shape)
y1_ev = 100*np.sin(x1_ev)
y2_ev = 100*np.sin(x2_ev)

plt.plot(x1_ev,y1_ev,'r.-')
plt.plot(x2_ev,y2_ev,'bo')
plt.title('Visualization of a function and some data points')
plt.show()'''