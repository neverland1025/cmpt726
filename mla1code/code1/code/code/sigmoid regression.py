"""Basic code for assignment 1."""
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,10]
#x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
train_err = dict()
test_err = dict()


train_bigfai=a1.design_matrix('sigmoid',x_train,0,0)
(w,train_error)=a1.linear_regression(x,t_train,train_bigfai,-1,0,100,2000)

test_bigfai=a1.design_matrix('sigmoid',x_test,0,0)
test_error=a1.evaluate_regression(test_bigfai,w,t_test)


print(train_error)


print(test_error)



#create a plot
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
x_ev = np.transpose(np.asmatrix(x_ev))

bigfai=a1.design_matrix('sigmoid',x_ev,0,0)
y_ev = np.transpose(w)*np.transpose(bigfai)



plt.plot(x_train,t_train,'bo')
plt.plot(x_test,t_test,'go')
plt.plot(x_ev,np.transpose(y_ev),'r.-')

plt.show()


