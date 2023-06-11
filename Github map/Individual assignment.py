import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# getting the temperatures
temperatures = pd.read_csv('HadCRUT.5.0.1.0.analysis.summary_series.global.annual.csv')
temperatures = temperatures.drop(columns = ['Lower confidence limit (2.5%)', 'Upper confidence limit (97.5%)'])
temperatures = temperatures.set_axis(['year', 'temperature'], axis = 1)

#getting CO2
co2 = pd.read_csv('co2_annual_20221026.csv')
co2 = co2.drop(columns = 'co2 uncertainty(ppm)')

#getting CH4
ch4 = pd.read_csv('ch4_annual_20221026.csv')
ch4 = ch4.drop(columns = 'ch4 uncertainty(ppb)')

#getting N2O
n2o = pd.read_csv('n2o_annual_20221026.csv')
n2o = n2o.drop(columns = 'n2o uncertainty(ppb)')

#getting sea ice area northern hemisphere
NHarea = pd.read_excel('Sea_Ice_Index_Monthly_Data_by_Year_G02135_v3.0.xlsx', sheet_name = 'NH-Area')
NHarea = NHarea.iloc[1: -1]
#avg area
NHavgarea = NHarea.iloc[:, [0, 14]]
NHavgarea = NHavgarea.set_axis(['year', 'NH average area sea ice'], axis = 1)                                            
#min area
NHyears = NHarea.iloc[:, 0] 
NHminarea = NHarea.min(axis = 1)
NHminarea = pd.concat([NHyears, NHminarea], axis = 1)
NHminarea = NHminarea.set_axis(['year', 'NH minimum area sea ice'], axis = 1) 
#max area
NHmaxhelp = NHarea.drop(columns = 'Unnamed: 0')
NHmaxarea = NHmaxhelp.max(axis = 1)
NHmaxarea = pd.concat([NHyears, NHmaxarea], axis = 1)
NHmaxarea = NHmaxarea.set_axis(['year', 'NH maximum area sea ice'], axis = 1) 

#getting sea ice area southern hemisphere
SHarea = pd.read_excel('Sea_Ice_Index_Monthly_Data_by_Year_G02135_v3.0.xlsx', sheet_name = 'SH-Area')
SHarea = SHarea.iloc[1: -1]
#avg area
SHavgarea = SHarea.iloc[:, [0, 14]]
SHavgarea = SHavgarea.set_axis(['year', 'SH average area sea ice'], axis = 1)                                   
#min area
SHyears = SHarea.iloc[:, 0]
SHminarea = SHarea.min(axis = 1)
SHminarea = pd.concat([SHyears, SHminarea], axis = 1)
SHminarea = SHminarea.set_axis(['year', 'SH minimum area sea ice'], axis = 1)
#max area
SHmaxhelp = SHarea.drop(columns = 'Unnamed: 0')
SHmaxarea = SHmaxhelp.max(axis = 1)
SHmaxarea = pd.concat([SHyears, SHmaxarea], axis = 1)
SHmaxarea = SHmaxarea.set_axis(['year', 'SH maximum area sea ice'], axis = 1)

#Global GDP
GDPraw = pd.read_excel('Download-GDPcurrent-USD-all.xlsx')
GDP = GDPraw.iloc[[1, 10]]
GDP = GDP.transpose()
GDP = GDP.iloc[3:]
GDP = GDP.set_axis(['year', 'GDP (global)'], axis = 1)
GDP['GDP (global)'] = GDP['GDP (global)'].apply(lambda x: int(x))
GDP = GDP.reset_index(drop = True)

#combining it all into one dataset
dataset = [temperatures, co2, ch4, n2o, NHavgarea, NHminarea, NHmaxarea, SHavgarea, SHminarea, SHmaxarea, GDP]

#merge all DataFrames into one
rawdata = reduce(lambda left,right: pd.merge(left, right, on = ['year'], how = 'outer'), dataset)
rawdata.set_index('year', inplace = True)

#visualising some stuff
plt.plot(rawdata['temperature'], color = 'red') # note that most plots are only legible when these lines are ran individually
plt.plot(rawdata['co2 global mean(ppm)'])
plt.plot(rawdata['ch4 global mean(ppb)'])
plt.plot(rawdata['n2o global mean(ppb)'])
plt.plot(rawdata['NH average area sea ice'])
plt.plot(rawdata['NH minimum area sea ice'])
plt.plot(rawdata['NH maximum area sea ice'])
plt.plot(rawdata['SH average area sea ice'])
plt.plot(rawdata['SH minimum area sea ice'])
plt.plot(rawdata['SH maximum area sea ice'])
plt.plot(rawdata['GDP (global)'])

#now lets visualise some comparisons
fewdata = rawdata.dropna()

'''RUN AS ONE BLOCK FROM HERE'''
fig, ax1 = plt.subplots()
ax1.plot(fewdata.index, fewdata['temperature'], color = 'tab:red')
#by selecting one of the ax2 code lines, said variable can be compared to the temperature
#this can only be done if the entire section is ran at once
ax2 = ax1.twinx()
#ax2.plot(fewdata.index, fewdata['co2 global mean(ppm)'], color='tab:blue')
#ax2.plot(fewdata.index, fewdata['ch4 global mean(ppm)'], color='tab:blue')
#ax2.plot(fewdata.index, fewdata['n2o global mean(ppm)'], color='tab:blue')
#ax2.plot(fewdata.index, fewdata['NH average area sea ice'], color='tab:blue')
#ax2.plot(fewdata.index, fewdata['NH minimum area sea ice'], color='tab:blue')
#ax2.plot(fewdata.index, fewdata['NH maximum area sea ice'], color='tab:blue')
#ax2.plot(fewdata.index, fewdata['SH average area sea ice'], color='tab:blue')
#ax2.plot(fewdata.index, fewdata['SH minimum area sea ice'], color='tab:blue')
#ax2.plot(fewdata.index, fewdata['SH maximum area sea ice'], color='tab:blue')
#ax2.plot(fewdata.index, fewdata['GDP (global)'], color='tab:blue')
'''TO HERE'''

#finding out what types of relationships there are in this dataset (hint: they're all linear)
sns.regplot(x = 'co2 global mean(ppm)', y = 'temperature', data = rawdata)      #linear
sns.regplot(x = 'ch4 global mean(ppb)', y = 'temperature', data = rawdata)      #linear
sns.regplot(x = 'n2o global mean(ppb)', y = 'temperature', data = rawdata)      #linear
sns.regplot(x = 'NH average area sea ice', y = 'temperature', data = rawdata)   #linear
sns.regplot(x = 'NH minimum area sea ice', y = 'temperature', data = rawdata)   #linear
sns.regplot(x = 'NH maximum area sea ice', y = 'temperature', data = rawdata)   #linear? low corr
sns.regplot(x = 'SH average area sea ice', y = 'temperature', data = rawdata)   #linear? low corr
sns.regplot(x = 'SH minimum area sea ice', y = 'temperature', data = rawdata)   #linear? low corr
sns.regplot(x = 'SH maximum area sea ice', y = 'temperature', data = rawdata)   #linear? low corr
sns.regplot(x = 'GDP (global)', y = 'temperature', data = rawdata)              #linear

#correlation plot
correlation = rawdata.corr()
heatmap = sns.heatmap(correlation, cmap = 'coolwarm', annot = False)

#normalizing the data
normaldata = fewdata.copy()
columns = normaldata.columns
indep_var = columns[1:]
normaldata[indep_var] = StandardScaler().fit_transform(normaldata[indep_var])

#making the training and test sets
y = normaldata['temperature']
x = normaldata.drop(columns = 'temperature')
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

#modelling - linear regression
LR = LinearRegression()
LR.fit(xtrain, ytrain)
LRpred = LR.predict(xtest)

#modelling - decision tree regression
DT = DecisionTreeRegressor()
DT.fit(xtrain, ytrain)
DTpred = DT.predict(xtest)

#modelling - random forests
RF = RandomForestRegressor()
RF.fit(xtrain, ytrain)
RFpred = RF.predict(xtest)

#modelling - ridge
#finding the best value for alpha using grid search
testridge = Ridge()
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = {'alpha': alphas}
searchridge = GridSearchCV(testridge, param_grid, cv = 6)
searchridge.fit(xtrain, ytrain)
# setting that alpha for the real ridge regressor
ridge = searchridge.best_estimator_
ridge.fit(xtrain, ytrain)
ridgepred = ridge.predict(xtest)

#modelling - lasso
#finding the best value for alpha using grid search
testlasso = Lasso()
searchlasso = GridSearchCV(testlasso, param_grid, cv = 6)
searchlasso.fit(xtrain, ytrain)
# setting that alpha for the real ridge regressor
lasso = searchlasso.best_estimator_
lasso.fit(xtrain, ytrain)
lassopred = lasso.predict(xtest)

#evaluation
#first a dataframe is generated that compares all predictions against the real test set
predictions = pd.DataFrame({
    'real value' : ytest,
    'LR prediction' : LRpred,
    'DT prediction' : DTpred,
    'RF prediction' : RFpred,
    'Ridge prediction' : ridgepred,
    'Lasso prediction' : lassopred})
predictions.reset_index(inplace = True)
#collecting the r-squared and the RMSE for all models
#LR
r2_LR = r2_score(ytest, LRpred)
RMSE_LR = np.sqrt(mean_squared_error(ytest, LRpred))
#DT
r2_DT = r2_score(ytest, DTpred)
RMSE_DT = np.sqrt(mean_squared_error(ytest, DTpred))
#RF
r2_RF = r2_score(ytest, RFpred)
RMSE_RF = np.sqrt(mean_squared_error(ytest, RFpred))
#ridge
r2_ridge = r2_score(ytest, ridgepred)
RMSE_ridge = np.sqrt(mean_squared_error(ytest, ridgepred))
#lasso
r2_lasso = r2_score(ytest, lassopred)
RMSE_lasso = np.sqrt(mean_squared_error(ytest, lassopred))

#combining them into one dataframe
index_v = ['Linear regression', 'Decision tree regression', 'Random forest regression', 'Ridge regression', 'Lasso regression']
index_h = ['R squared', 'RMSE']
data = [[r2_LR, RMSE_LR],
        [r2_DT, RMSE_DT],
        [r2_RF, RMSE_RF],
        [r2_ridge, RMSE_ridge],
        [r2_lasso, RMSE_lasso]]
performance = pd.DataFrame(data, index = index_v, columns = index_h)

#finally, converting them to csv files for PowerBi visualization
predictions.to_csv('model predictions.csv', index = False)
performance.to_csv('model performance.csv', index = True)
