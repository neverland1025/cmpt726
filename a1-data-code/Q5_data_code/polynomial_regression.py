#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]




degree=6

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


#calculate from degree 1 to 6


for n in range(1, degree + 1):
    # calculate theta and w*
    print(x_train)
    train_bigfai= func_x_times(x_train, n)

    w = np.linalg.pinv(train_bigfai) * t_train

    y = np.transpose(w) * np.transpose(train_bigfai)
    train_error=t_train-np.transpose(y)

    rms_train=np.sqrt(np.mean(np.square(train_error)))

    test_bigfai=func_x_times(x_test,n)
    y_test=np.transpose(w) * np.transpose(test_bigfai)
    test_error=t_test-np.transpose(y_test)
    rms_test=np.sqrt(np.mean(np.square(test_error)))



    train_err[n] = rms_train
    test_err[n] = rms_test







# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions

# (w, tr_err) = a1.linear_regression()
# (t_est, te_err) = a1.evaluate_regression()








# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
plt.plot([float(k) for k in train_err.keys()],[float(v) for v in train_err.values()])
plt.plot([float(k) for k in test_err.keys()],[float(v) for v in test_err.values()])
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Training error','Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
