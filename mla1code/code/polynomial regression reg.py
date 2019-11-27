import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
lambda_list=[0,0.01,0.1,1,10,100,1000,10000]
validation_err_avg=dict()

def regression_reg(lamda,deg):
    validation_err=0
    for i in range(0,10):
        print(i)
        rms_val_error=0
        x_val=x_train[i*10:(i+1)*10,:]
        t_val=t_train[i*10:(i+1)*10,:]
        x_train_use=np.concatenate((x_train[0:i*10,:],x_train[(i+1)*10:,:]),0)
        t_train_use=np.concatenate((t_train[0:i*10,:],t_train[(i+1)*10:,:]),0)

        bigphi = a1.design_matrix('polynomial', x_train_use, deg, 1)

        (w, rms_train) = a1.linear_regression(x_train_use, t_train_use, bigphi,lamda,deg, 0, 0)
        #print(w)


        bigfai_val=a1.design_matrix('polynomial', x_val, deg, 1)
        rms_val_error=a1.evaluate_regression(bigfai_val, w, t_val)
        #print(rms_val_error)

        validation_err += rms_val_error
        #print(validation_err)
    validation_err_avg[lamda] = validation_err / 10
    print(validation_err_avg)


regression_reg(0, 2)
regression_reg(0.01, 2)
regression_reg(0.1, 2)
regression_reg(1, 2)
regression_reg(10, 2)
regression_reg(100, 2)
regression_reg(1000, 2)
regression_reg(10000, 2)





label = sorted(validation_err_avg.keys())
error = []
for key in label:
    error.append(validation_err_avg[key])
print(error)
plt.semilogx(label, error)
plt.ylabel('Average RMS')
plt.legend(['Average Validation error'])

plt.show()

