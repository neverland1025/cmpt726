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





train_err1 = dict()
test_err1 = dict()

train_err = dict()
test_err = dict()


#with bias


for i in range(0,8):
    train_feature=x_train[:,i]
    test_feature = x_test[:,i]
    bigphi=a1.design_matrix('polynomial',train_feature,3,1)

    (w,rms_train)=a1.linear_regression(train_feature,t_train,bigphi,-1,3,0,0)

    test_phi=a1.design_matrix('polynomial',test_feature,3,1)
    rms_test=a1.evaluate_regression(test_phi,w,t_test)
    train_err[i] = rms_train
    test_err[i] = rms_test

#without bias
for i in range(0,8):
    train_feature=x_train[:,i]
    test_feature = x_test[:,i]
    bigphi=a1.design_matrix('polynomial',train_feature,3,0)

    (w,rms_train)=a1.linear_regression(train_feature,t_train,bigphi,-1,3,0,0)

    test_phi=a1.design_matrix('polynomial',test_feature,3,0)
    rms_test=a1.evaluate_regression(test_phi,w,t_test)
    train_err[i] = rms_train
    test_err[i] = rms_test


print(train_err)
print(test_err)

    #print(set)
    #print("over")








# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions

# (w, tr_err) = a1.linear_regression()
# (t_est, te_err) = a1.evaluate_regression()








# Produce a plot of results.
plt.rcParams.update({'font.size': 15})

plt.bar([float(k)+1 for k in train_err.keys()],[float(v) for v in train_err.values()],width=0.2)
plt.bar([float(k)+1.2 for k in test_err.keys()],[float(v) for v in test_err.values()],width=0.2)
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Training error','Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
#x_ticks=[i+0.2 for i in train_err.keys()]
plt.xticks()
plt.show()
