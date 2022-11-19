# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:11:28 2020

@author: AmirAli Kalbasi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %% import data
Rataddress = ['Rat1.xlsx','Rat1','Rat1 NAc','Rat1 Hip',
              ...
              'Rat10.xlsx','Rat10','Rat10 NAc','Rat10 Hip',
             ]
Rat_number =1
dataset = pd.read_excel(Rataddress[Rat_number,0]) 
# %% start
# Data: 1- HIP
#       2- NAc)
size_data = np.max(np.shape(dataset))
test_number = np.int(np.round(size_data/10))
X_train, x_test = dataset[0:-test_number], dataset[-test_number:]
print(X_train.shape) 
print(x_test.shape)

transform_data = X_train.diff().dropna()
transform_data.describe()

# %% Select the Order (P) of VAR model
# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
    
model = VAR(transform_data)
for i in range(0,50):
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

model_fitted = model.fit(4)
def adjust(val, length= 2): return str(val).ljust(length)
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(dataset.columns, out):
    print(adjust(col), ':', round(val, 2))
    
# %% Granger causality tests
from statsmodels.tsa.stattools import grangercausalitytests 
maxlag=50 
test = 'ssr_chi2test' 
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    X_train = pd.DataFrame (np.zeros((len (variables), len(variables))), columns=variables, index=variables) 
    for c in X_train.columns:
        for r in X_train.index:
            test_result = grangercausalitytests (data[[r, c]], maxlag=maxlag, verbose=False) 
            p_values = [round(test_result[i+1][0][test][1],4) for i in range (maxlag)] 
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}') 
            min_p_value = np.min(p_values)
            X_train.loc[r, c] = min_p_value 
    print(f'{Rataddress[Rat_number,1]} : ')
    X_train.columns = [var + '_x' for var in variables]
    X_train.index = [var + '_y' for var in variables] 
    return X_train

grangers_causation_matrix(X_train, variables = X_train.columns)
# %%

from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(transform_data, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(transform_data,-1,5) 
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)
    print('Name :: Test Stat > C(95%) => Signif \n', '--'*20)
    for col, trace, cvt in zip(transform_data.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' => ', trace > cvt)

cointegration_test(X_train)

# %%
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = transform_data.values[-lag_order:]

import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR
mod = smt.VAR(transform_data)
res = mod.fit(maxlags=lag_order, ic='aic')
print(res.summary())



from statsmodels.stats.stattools import durbin_watson 
out = durbin_watson(res.resid)
for col, val in zip(transform_data.columns, out):
    print((col), ':', round(val, 2))

    
pred= res. forecast(y=forecast_input, steps=test_number) 
pred_df = pd.DataFrame (pred, index=dataset.index[-test_number:], columns=dataset.columns)

pred_inverse = pred_df.cumsum() # reversing difference 
f = pred_inverse + x_test # inverse the differece values 
print(f)


plt.figure(figsize= (12,5))
plt.xlabel('date')
ax1 = x_test.NAc.plot(color='blue', grid = True, label = 'Actual NAc') 
ax2 = f.NAc.plot(color='red', grid = True, secondary_y=True, label = 'Predicted NAc')
ax1.legend (loc=1) 
ax2.legend (loc=2) 
plt.title('Predicted Vs Actual NAc')


plt.figure(figsize= (12,5))
plt.xlabel('date')
ax1 = x_test.Hip.plot(color='blue', grid = True, label = 'Actual Hip') 
ax2 = f.Hip.plot(color='red', grid = True, secondary_y=True, label = 'Predicted Hip')
ax1.legend (loc=1) 
ax2.legend (loc=2) 
plt.title('Predicted Vs Actual Hip')

res.plot_acorr()

from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_squared_error 
from math import sqrt
forecast_errors = [x_test.NAc[f.NAc.index[i]]-f.NAc[f.NAc.index[i]] for i in range(len(x_test.NAc))] 
bias = sum(forecast_errors) * 1.0/len(x_test.NAc) 
print(f'Bias: %{bias}')
mae = mean_absolute_error(x_test.NAc, f.NAc) 
print(f'MAE: %{mae}')
mse = mean_squared_error(x_test.NAc, f.NAc) 
print(f'MSE: %{mse}')

plt.figure(figsize= (30,10))
ax1 =X_train.NAc[-5000:].plot( color='blue', grid = True,label = 'Train NAc')
ax2 =x_test.NAc.plot( color='green', grid = True,label = 'Actual NAc')
ax3 =f.NAc.plot( style='.',color='red', grid = True,label = 'Predict NAc')
ax1.legend (loc=1) 
ax2.legend (loc=2) 
ax3.legend (loc=3)
plt.title(f'{Rataddress[Rat_number,1]} : Predicted Vs Actual NAc')


plt.figure(figsize= (30,10))
ax1 =X_train.Hip[-5000:].plot( color='blue', grid = True,label = 'Train Hip')
ax2 =x_test.Hip.plot( color='green', grid = True,label = 'Actual Hip')
ax3 =f.Hip.plot( style='.',color='red', grid = True,label = 'Predict Hip')
ax1.legend (loc=1) 
ax2.legend (loc=2) 
ax3.legend (loc=3)
plt.title(f'{Rataddress[Rat_number,1]} : Predicted Vs Actual Hip')

