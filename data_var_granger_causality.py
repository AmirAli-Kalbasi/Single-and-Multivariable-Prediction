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
Rataddress = ['C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat1/Rat1_natural_Pre.xlsx','Rat1 Natural Pre','Rat1 Natural Pre NAc','Rat1 Natural Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat1/Rat1_natural_Post.xlsx','Rat1 Natural Post','Rat1 Natural Post NAc','Rat1 Natural Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat7/Rat7_natural_Pre.xlsx','Rat7 Natural Pre','Rat7 Natural Pre NAc','Rat7 Natural Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat7/Rat7_natural_Post.xlsx','Rat7 Natural Post','Rat7 Natural Post NAc','Rat7 Natural Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat8/Rat8_natural_Pre.xlsx','Rat8 Natural Pre','Rat8 Natural Pre NAc','Rat8 Natural Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat8/Rat8_natural_Post.xlsx','Rat8 Natural Post','Rat8 Natural Post NAc','Rat8 Natural Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat11/Rat11_natural_Pre.xlsx','Rat11 Natural Pre','Rat11 Natural Pre NAc','Rat11 Natural Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat11/Rat11_natural_Post.xlsx','Rat11 Natural Post','Rat11 Natural Post NAc','Rat11 Natural Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat13/Rat13_natural_Pre.xlsx','Rat13 Natural Pre','Rat13 Natural Pre NAc','Rat13 Natural Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat13/Rat13_natural_Post.xlsx','Rat13 Natural Post','Rat13 Natural Post NAc','Rat13 Natural Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat15/Rat15_natural_Pre.xlsx','Rat15 Natural Pre','Rat15 Natural Pre NAc','Rat15 Natural Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat15/Rat15_natural_Post.xlsx','Rat15 Natural Post','Rat15 Natural Post NAc','Rat15 Natural Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat16/Rat16_natural_Pre.xlsx','Rat16 Natural Pre','Rat16 Natural Pre NAc','Rat16 Natural Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(2)_natural/Rat16/Rat16_natural_Post.xlsx','Rat16 Natural Post','Rat16 Natural Post NAc','Rat16 Natural Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat1/Rat1_morphine_Pre.xlsx','Rat1 Morphine Pre','Rat1 Morphine Pre NAc','Rat1 Morphine Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat1/Rat1_morphine_Post.xlsx','Rat1 Morphine Post','Rat1 Morphine Post NAc','Rat1 Morphine Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat3/Rat3_morphine_Pre.xlsx','Rat3 Morphine Pre','Rat3 Morphine Pre NAc','Rat3 Morphine Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat3/Rat3_morphine_Post.xlsx','Rat3 Morphine Post','Rat3 Morphine Post NAc','Rat3 Morphine Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat5/Rat5_morphine_Pre.xlsx','Rat5 Morphine Pre','Rat5 Morphine Pre NAc','Rat5 Morphine Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat5/Rat5_morphine_Post.xlsx','Rat5 Morphine Post','Rat5 Morphine Post NAc','Rat5 Morphine Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat6/Rat6_morphine_Pre.xlsx','Rat6 Morphine Pre','Rat6 Morphine Pre NAc','Rat6 Morphine Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat6/Rat6_morphine_Post.xlsx','Rat6 Morphine Post','Rat6 Morphine Post NAc','Rat6 Morphine Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat7/Rat7_morphine_Pre.xlsx','Rat7 Morphine Pre','Rat7 Morphine Pre NAc','Rat7 Morphine Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat7/Rat7_morphine_Post.xlsx','Rat7 Morphine Post','Rat7 Morphine Post NAc','Rat7 Morphine Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat8/Rat8_morphine_Pre.xlsx','Rat8 Morphine Pre','Rat8 Morphine Pre NAc','Rat8 Morphine Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(4)_morphine/Rat8/Rat8_morphine_Post.xlsx','Rat8 Morphine Post','Rat8 Morphine Post NAc','Rat8 Morphine Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat9/Rat9_saline_Pre.xlsx','Rat9 Saline Pre','Rat9 Saline Pre NAc','Rat9 Saline Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat9/Rat9_saline_Post.xlsx','Rat9 Saline Post','Rat9 Saline Post NAc','Rat9 Saline Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat10/Rat10_saline_Pre.xlsx','Rat10 Saline Pre','Rat10 Saline Pre NAc','Rat10 Saline Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat10/Rat10_saline_Post.xlsx','Rat10 Saline Post','Rat10 Saline Post NAc','Rat10 Saline Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat11/Rat11_saline_Pre.xlsx','Rat11 Saline Pre','Rat11 Saline Pre NAc','Rat11 Saline Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat11/Rat11_saline_Post.xlsx','Rat11 Saline Post','Rat11 Saline Post NAc','Rat11 Saline Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat12/Rat12_saline_Pre.xlsx','Rat12 Saline Pre','Rat12 Saline Pre NAc','Rat12 Saline Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat12/Rat12_saline_Post.xlsx','Rat12 Saline Post','Rat12 Saline Post NAc','Rat12 Saline Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat13/Rat13_saline_Pre.xlsx','Rat13 Saline Pre','Rat13 Saline Pre NAc','Rat13 Saline Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat13/Rat13_saline_Post.xlsx','Rat13 Saline Post','Rat13 Saline Post NAc','Rat13 Saline Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat14/Rat14_saline_Pre.xlsx','Rat14 Saline Pre','Rat14 Saline Pre NAc','Rat14 Saline Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat14/Rat14_saline_Post.xlsx','Rat14 Saline Post','Rat14 Saline Post NAc','Rat14 Saline Post Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat15/Rat15_saline_Pre.xlsx','Rat15 Saline Pre','Rat15 Saline Pre NAc','Rat15 Saline Pre Hip',
'C:/Users/AmirAli/OneDrive/Desktop/LFP_CPP _edited/(6)_saline/Rat15/Rat15_saline_Post.xlsx','Rat15 Saline Post','Rat15 Saline Post NAc','Rat15 Saline Post Hip']
Rataddress=np.reshape(Rataddress,[40,4])
number_of_rats = np.shape(Rataddress[:,0])[0]

Rat_number =1
dataset1 = pd.read_excel(Rataddress[Rat_number,0]) 
# %% start
#size_data=np.where(dataset1.Chamber!=dataset1.Chamber[0])[0][0]-1
dataset=dataset1[200000:300000]    
dataset=dataset.rename(columns={"Nac": "NAc", "Hippocampus": "Hip", "Chamber": "Chambers"})
size_data = np.max(np.shape(dataset))
# %% visualization data

sns.lmplot('NAc','Hip',dataset,hue='Chambers',fit_reg=False)
ax = plt.gca()
ax.set_title(f'{Rataddress[Rat_number,1] }')

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# Plot
fig, axes = plt.subplots(nrows=3, ncols=1, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = dataset[dataset.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(f'{Rataddress[Rat_number,1]} ' + dataset.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_alpha(0)
    ax.tick_params(labelsize=6)
    plt.tight_layout()

# %% check distribution of data
from scipy import stats
stat,p = stats.normaltest(dataset.NAc)
print(f'{Rataddress[Rat_number,1]} : statastics={stat} , p={p}')
alpha = 0.5
if p>alpha:
    print(f'{Rataddress[Rat_number,1]} looks Gaussian(fail to reject H0)')
else:
    print(f'{Rataddress[Rat_number,1]} do not look Gaussian(reject H0)')
    
print(f'{Rataddress[Rat_number,1]} : Kurtosis of normal distribution: {stats.kurtosis(dataset.NAc)}') 
print(f'{Rataddress[Rat_number,1]} : Skewness of normal distribution: {stats.skew(dataset.NAc)}')

print(f'{Rataddress[Rat_number,1]} : Kurtosis of normal distribution: {stats.kurtosis(dataset.Hip)}') 
print(f'{Rataddress[Rat_number,1]} : Skewness of normal distribution: {stats.skew(dataset.Hip)}')



plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
dataset['NAc'].hist(bins=50)
plt.title(f'NAc {Rataddress[Rat_number,1]}')
plt.subplot(1,2,2)
stats.probplot(dataset['NAc'], plot=plt)
ax = plt.gca()
ax.set_title(f"Probplot plot {Rataddress[Rat_number,1]} NAc")
dataset.NAc.describe().T

plt.figure(figsize=(14,6))
corr = dataset.corr() 
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12}) 
heat_map=plt.gcf()
heat_map.set_size_inches (15,10)
plt.xticks (fontsize=10)
plt.yticks (fontsize=10)
plt.title(f'{Rataddress[Rat_number,1]}')
plt.show()

# %% Drop Chamber
dataset = dataset.drop(['Chambers'], axis=1)
# %% Stationary check
test_number = np.int(np.round(size_data/10))
X_train, x_test = dataset[0:-test_number], dataset[-test_number:]
print(X_train.shape) 
print(x_test.shape)

transform_data = X_train.diff().dropna()
transform_data.describe()

from statsmodels.tsa.stattools import adfuller
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller (series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]} 
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)
    print(f' {Rataddress[Rat_number,1]} : Augmented Dickey-Fuller Test on "{name}"', "\n ", '-'*47) 
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.') 
    print(f' significance Level = {signif}')
    print(f' Test Statistic = {output["test_statistic"]}') 
    print(f' No. Lags chosen = {output["n_lags"]}')
    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')
    if p_value <= signif:
        print(f" => P-value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => {Rataddress[Rat_number,1]} {name}  is stationary.") 
    else:
        print(f" => P-value = {p_value}. Weak evidence to reject the Null Hypothesis.") 
        print(f" => {Rataddress[Rat_number,1]} {name} is Non-Stationary.")

for name, column in dataset.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')    


fig, axes = plt.subplots(nrows=2, ncols=1, dpi=120, figsize=(10,6)) 
for i, ax in enumerate(axes.flatten()):
    data = transform_data[transform_data.columns[i]] 
    ax.plot(data, color='red', linewidth=1)  
    ax.set_title(f'Transform {Rataddress[Rat_number,1]} ' + dataset.columns[i])
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position ('none') 
    ax.spines ["top"].set_alpha (0) 
    ax.tick_params (labelsize=6)
plt. tight_layout()


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

