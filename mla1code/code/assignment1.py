"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    return (x - mvec)/stdvec

def sigmoid(x,u,s):
    return 1/(1+np.exp((u-x)/s))



def linear_regression(x, t, phi, reg_lambda, deg, mu, s):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function

            
    # Learning Coefficients
    if (reg_lambda >0) or (reg_lambda==0):
        # regularized regression
        print("yes")

        secondtimes = np.transpose(phi) * phi

        I = np.eye(secondtimes.shape[0])

        w = np.linalg.inv(reg_lambda * I + secondtimes) * np.transpose(phi) * t
        rms_train=None
    else:
        # no regularization

        w = np.linalg.pinv(phi) * t

        y = np.transpose(w) * np.transpose(phi)
        train_err = t - np.transpose(y)
        rms_train = np.sqrt(np.mean(np.square(train_err)))



    # Measure root mean squared error on training data.
    #train_err = None

    return (w, rms_train)



def design_matrix(basis,martrix,deg,bias):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
        ?????

    Returns:
      phi design matrix
    """

    if basis == 'polynomial' and bias==1:
        x_zero_time = np.ones(martrix.shape[0], dtype=int)

        x_zero_transpose = x_zero_time.reshape(x_zero_time.shape[0], 1)

        new_martrix = martrix
        martrix = np.concatenate((x_zero_transpose, martrix), 1)
        if deg == 1:
            return martrix
        else:

            for n in range(2, deg + 1):
                x_times = np.power(new_martrix, n)
                martrix = np.concatenate((martrix, x_times), 1)

            return martrix
    elif basis == 'polynomial' and bias==0:
        new_martrix = martrix

        if deg == 1:
            return martrix
        else:

            for n in range(2, deg + 1):
                x_times = np.power(new_martrix, n)
                martrix = np.concatenate((martrix, x_times), 1)

            return martrix
    elif basis == 'sigmoid':
        martrix1 = martrix
        x_bias = np.ones(martrix.shape[0], dtype=int)
        # print(x_bias)

        x_bias_transpose = x_bias.reshape(x_bias.shape[0], 1)
        # print(x_bias_transpose)
        u1 = np.apply_along_axis(sigmoid, 0, martrix, 100, 2000)
        # print(u1)
        # print(u1.shape)
        u1_trans = u1.reshape(u1.shape[1], 1)
        # print(u1_trans)
        new_martrix = np.concatenate((x_bias_transpose, u1_trans), 1)
        # print("new_martix")
        # print(new_martrix)
        u2 = np.apply_along_axis(sigmoid, 0, martrix, 10000, 2000)
        # print("u2")
        # print(u2)
        u2_trans = u2.reshape(u2.shape[1], 1)
        martrix2 = np.concatenate((new_martrix, u2_trans), 1)
        # print(martrix2)
        return martrix2

    else: 
        assert(False), 'Unknown basis %s' % basis



def evaluate_regression(phi,w,t_test):
    """Evaluate linear regression on a dataset.

    Args:
      ?????

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None

      """
    #test_bigfai = func_x_times(x_test, n)
    y_test = np.transpose(w) * np.transpose(phi)
    test_error = t_test - np.transpose(y_test)
    rms_test = np.sqrt(np.mean(np.square(test_error)))

    return (rms_test)
