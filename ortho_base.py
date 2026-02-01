import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.special import legendre
from sympy import symbols, diff, exp, factorial
from sympy import symbols, integrate, sqrt, laguerre, lambdify
from sympy import symbols, diff, exp, factorial
from sklearn.metrics import mean_squared_error
import math

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# rmse
def rmse_cal(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# mse
def mse_cal(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

# laguerre_polynomial
def laguerre_polynomial(n, x0):
    x0 = symbols('x0')
    term = exp(x0) * diff((x0**n * exp(-x0)), x0, n)
    return term

# hermite_polynomial
def hermite_polynomial(n, x):
    x = symbols('x')
    term = (-1)**n * exp(x**2) * diff(exp(-x**2), x, n)
    return term

#Legendre polynomial expansion
def lefun(degree,x):
    legendre_poly = legendre(degree)
    legendre_values = np.polyval(legendre_poly, x)
    lefun_value = legendre_values
    return lefun_value

#Lagrange expansion
def lafun(degree, x_value):
    x0 = symbols('x0')
    laguerre_poly = laguerre_polynomial(degree, x0)
    laguerre_poly_fun = lambdify(x0, laguerre_poly, 'numpy')
    if degree == 0:
        lafun_value = np.array([1] * len(x_value))
    else:
        lafun_value = laguerre_poly_fun(x_value)
    return lafun_value

#Hermite expansion
def hefun(degree,x):
    x0 = symbols('x0')
    hermite_poly = hermite_polynomial(degree, x0)
    hermite_values = np.polynomial.hermite.hermval(x, [0] * degree + [1])
    hefun_value = hermite_values
    return hefun_value

def akfun(degree,x):
    if min(x)>=-1 and max(x)<=1:
        value = lefun(degree,x)
        print("Using Legendre orthogonal polynomials")
    elif min(x)>0 and max(x)>1:
        #laguerre
        # laguerre_poly = laguerre_polynomial(degree, x)
        value = lafun(degree, x)
        print("Using Lagrange orthogonal polynomials")
    elif min(x)<-1 and max(x)>1:
        #hermite
        # hermite_poly = hermite_polynomial(degree, x)
        value = hefun(degree, x)
        print("Using Hermite orthogonal polynomials")
    return value

# Orthogonal series expansion for different x
def ak(k,x):
    f1 = []
    for i in range(0,k):
        fvalue = akfun(i,x).tolist()
        f1.append(fvalue)
    return f1

def sx_funcation(x):
    if min(x) >= -1 and max(x) <= 1:
        print("Using Legendre orthogonal polynomials")
    elif min(x) > 0 and max(x) > 1:
        print("Using Lagrange orthogonal polynomials")
    elif min(x) < -1 and max(x) > 1:
        print("Using Hermite orthogonal polynomials(0,k-1)")



def standard_normal_pdf(x):
    """
    Compute the value of the standard normal distribution PDF at x.
    """
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-x**2 / 2)

def compute_sum(x_values):
    """
    Compute the sum of f(x_t) for a list of x_t values.
    """
    return sum(standard_normal_pdf(x) for x in x_values)

def index_of_agreement(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Mean of actual values
    mean_actual = np.mean(actual)
    
    # Numerator: Sum of squared errors
    numerator = np.sum((predicted - actual) ** 2)
    
    # Denominator: Sum of squared errors based on actual values
    denominator = np.sum((np.abs(predicted - mean_actual) + np.abs(actual - mean_actual)) ** 2)
    
    # Consistency Index
    d = 1 - numerator / denominator if denominator != 0 else 0
    return d

def calculate_IA(f, f_hat):
    f = np.array(f)
    f_hat = np.array(f_hat)
    # Calculate the mean of observed and predicted values
    f_mean = np.mean(f)
    f_hat_mean = np.mean(f_hat)
    # Calculate the numerator part：sum((f_n - f_hat_n)^2)
    numerator = np.sum((f - f_hat) ** 2)
    # Calculate the denominator part：sum((|f_hat_n - f_hat_mean| + |f_n - f_mean|)^2)
    denominator = np.sum((np.abs(f_hat - f_hat_mean) + np.abs(f - f_mean)) ** 2)
    # Calculate IA
    IA = 1 - (numerator / denominator)
    return IA

def calculate_U1(f, f_hat):
    """
    calculate_U1

    Parameters：
    f (array-like): sequence of observed values
    f_hat (array-like): sequence of predicted values

    return：
    float: U1 value
    """

    f = np.array(f)
    f_hat = np.array(f_hat)
    
    numerator = np.sqrt(np.mean((f - f_hat) ** 2))
    denominator = np.sqrt(np.mean(f ** 2)) + np.sqrt(np.mean(f_hat ** 2))

    U1 = numerator / denominator
    
    return U1

def mann_whitney_u(x, y):
    x = np.array(x)
    y = np.array(y)
    combined = np.concatenate([x, y])
    ranks = rankdata(combined)  

    ranks_x = ranks[:len(x)]  
    r_x = np.sum(ranks_x)  

    n1, n2 = len(x), len(y)
    u1 = r_x - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    
    return u1, u2
