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
        x_val=x_train[i*10:(i+1)*10,:]
        t_val=t_train[i*10:(i+1)*10,:]
        x_train_use=np.concatenate((x_train[0:i*10,:],x_train[(i+1)*10:,:]),0)
        t_train_use=np.concatenate((t_train[0:i*10,:],t_train[(i+1)*10:,:]),0)

        bigfai=a1.func_x_times(x_train_use,deg)
        secondtimes=np.transpose(bigfai)*bigfai

        I=np.eye(secondtimes.shape[0])

        w=np.linalg.inv(lamda*I+secondtimes)*np.transpose(bigfai)*t_train_use

        bigfai_val=a1.func_x_times(x_val,2)
        y_val=np.transpose(w)*np.transpose(bigfai_val)
        t_val_error = t_val - np.transpose(y_val)
        rms_val_error = np.sqrt(np.mean(np.square(t_val_error)))
        validation_err += rms_val_error
    validation_err_avg[lamda] = validation_err / 10


regression_reg(0, 2)  # replace lambda=0 with lambda=10^-5 for plotting purpose.
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

