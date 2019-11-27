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
#print(x_train)
def sigmoid(x,u,s):
    return 1/(1+np.exp((u-x)/s))

def sig(martrix,u,s):
    sig=np.apply_along_axis(sigmoid, 0, martrix, u, s)


    return sig
#create a phi with bias
def sig_phi_bias(martrix):
    martrix1=martrix
    x_bias = np.ones(martrix.shape[0], dtype=int)
    #print(x_bias)

    x_bias_transpose = x_bias.reshape(x_bias.shape[0], 1)
    #print(x_bias_transpose)
    u1=sig(martrix,100,2000)
    #print(u1)
    #print(u1.shape)
    u1_trans=u1.reshape(u1.shape[1],1)
    #print(u1_trans)
    new_martrix = np.concatenate((x_bias_transpose, u1_trans), 1)
    #print("new_martix")
    #print(new_martrix)
    u2=sig(martrix1,10000,2000)
    #print("u2")
    #print(u2)
    u2_trans = u2.reshape(u2.shape[1], 1)
    martrix2 = np.concatenate((new_martrix,u2_trans),1)
    print(martrix2)
    #print(martrix2)
    return martrix2



train_bigfai = sig_phi_bias(x_train)
#print(train_bigfai)
w = np.linalg.pinv(train_bigfai) * t_train

y = np.transpose(w) * np.transpose(train_bigfai)
train_error = t_train - np.transpose(y)
rms_train = np.sqrt(np.mean(np.square(train_error)))

print(rms_train)

test_bigfai = sig_phi_bias(x_test)
y_test = np.transpose(w) * np.transpose(test_bigfai)
test_error = t_test - np.transpose(y_test)
rms_test = np.sqrt(np.mean(np.square(test_error)))
print(rms_test)



#create a plot
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
x_ev = np.transpose(np.asmatrix(x_ev))
# TO DO:: Put your regression estimate here in place of x_ev.
bigfai=sig_phi_bias(x_ev)
y_ev = np.transpose(w)*np.transpose(bigfai)

# Evaluate regression on the linspace samples.
#y_ev = np.random.random_sample(x_ev.shape)
#y_ev = 100*np.sin(x_ev)


plt.plot(x_train,t_train,'bo')
plt.plot(x_test,t_test,'go')
plt.plot(x_ev,np.transpose(y_ev),'r.-')
plt.legend(['Training data','Test data','Learned Function'])
plt.title('A visualization of a regression estimate using random outputs')
plt.show()


