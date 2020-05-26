import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import datetime as dt
from scipy import stats
import math
import operator

''' read data '''
data_ret = pd.read_pickle('data_ret_new.pkl')
df_negative = pd.read_pickle('df_negative_new_1.pkl')
df_negative = df_negative[df_negative.date > dt.datetime(2012,1,3)]
data_ret = data_ret[data_ret.date > dt.datetime(2012,1,3)]
data_ret.index= data_ret['date']
data_ret = data_ret.drop(columns = ['date'])
data_ret= data_ret.iloc[:,(data_ret.values < 1).all(axis=0) & (data_ret.values > -1).all(axis=0)]

stocks = data_ret.columns.unique().tolist()
stocks.remove('^GSPC')

''' parameter initialization '''
window = 10
estimation_period = 200
windows_indexes = range(-window, window + 1)
estimation_indexes = range(-estimation_period- window, - window)

dr_equities_window = pd.DataFrame(index=windows_indexes)
dr_equities_estimation = pd.DataFrame(index=estimation_indexes)
dr_market_window = pd.DataFrame(index=windows_indexes)
dr_market_estimation = pd.DataFrame(index=estimation_indexes)

''' for each stock, compute daily stock and market return during the window period'''
for symbol in stocks:
    ''' get the date on which negative event happens '''
    neg_event_dates = df_negative[df_negative.ticker == symbol].date.tolist()
    for neg_event in neg_event_dates:
        evt_idx = data_ret.index.get_loc(neg_event)
        col_name = symbol + ' ' + neg_event.strftime('%Y-%m-%d')
    
        ''' starting and ending index for the estimation period '''
        start_idx = evt_idx - window - estimation_period
        end_idx = evt_idx - window
        
        if start_idx < 0 or evt_idx + window >= data_ret.shape[0]:
            continue
    
        ''' compute daily stock and market return during the pre-event window '''
        new_dr_equities_estimation = data_ret[symbol][start_idx:end_idx]
        new_dr_equities_estimation.index = estimation_indexes
        dr_equities_estimation[col_name] = new_dr_equities_estimation
    
        new_dr_market_estimation = data_ret['^GSPC'][start_idx:end_idx]
        new_dr_market_estimation.index = estimation_indexes
        dr_market_estimation[col_name] = new_dr_market_estimation

        ''' starting and ending index for the window period '''
        start_idx = evt_idx - window 
        end_idx = evt_idx + window + 1
        
        ''' compute daily stock and market return during the window period '''
        new_dr_equities_window = data_ret[symbol][start_idx:end_idx]
        new_dr_equities_window.index = windows_indexes
        dr_equities_window[col_name] = new_dr_equities_window
                                                 
        new_dr_market_window = data_ret['^GSPC'][start_idx:end_idx]
        new_dr_market_window.index = windows_indexes
        dr_market_window[col_name] = new_dr_market_window
        
''' initalize regression_estimation and expected_return datasets '''
reg_estimation = pd.DataFrame(index=dr_market_estimation.columns,columns=['intercept', 'beta', 'rse'])
er = pd.DataFrame(index=dr_market_window.index,columns=dr_market_window.columns)

''' compute the expected return of each date using the regression'''
for col in dr_market_estimation.columns:
    x = dr_market_estimation[col]
    y = dr_equities_estimation[col]
    slope, intercept, _, _, _ = stats.linregress(x, y)
    reg_estimation['beta'][col] = slope
    reg_estimation['intercept'][col] = intercept
    reg_estimation['rse'][col] = sum((y - slope * x - intercept)**2)/(len(x)-2)
    er[col] = intercept + dr_market_window[col] * slope
    
er.columns.name = 'Expected return'
''' calculate abnormal return '''
ar = dr_equities_window - er
ar.columns.name = 'Abnormal return'

''' remove AR with NA or all values as 0 '''
ar = ar.dropna(axis = 1, how = 'any')
ar = ar.loc[:, (ar != 0).any(axis=0)]

''' calculate cumulative abnormal return '''
car = ar.apply(np.cumsum)
car.columns.name = 'Cum Abnormal Return'

''' calcualte mean for CAR and AR '''
mean_car = car.mean(axis=1)
mean_car.name = 'CAR'

mean_ar = ar.mean(axis=1)
mean_ar.name = 'AR'

''' compute mean and variance for CAAR '''
cols = car.columns.tolist()
rse = reg_estimation.loc[reg_estimation['rse'].index.isin(cols),'rse']
var_car = (window*2+1) * rse
mean_val = mean_car[window]
print('mean = %.5f'%(mean_val))

N = len(cols)
var_caar = sum(var_car)/(N*N)
t_car = mean_val/math.sqrt(var_caar)

pval = stats.t.sf(np.abs(t_car), N-1)*2
print('t-statistic = %6.3f, pvalue = %6.4f'%(t_car, pval))

''' plot CAR'''
x = mean_car.index.values
y = mean_car.values
z = mean_ar.values
fig, ax = plt.subplots(figsize=(8,5))
label = mean_car.name
ax.plot(x, y, label=label)
ax.set_xlabel('window')
ax.set_ylabel('CAR')
plt.title('cumulative abnormal return(CAR) within the event window')
plt.legend()


''' compute CAR by label '''
def car_by_label(label):
    ''' select ticker and date for negative event in this label category '''
    negative_by_label = df_negative[df_negative.label2 == label][['ticker', 'date']]
    negative_by_label['date_str'] = negative_by_label['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    negative_by_label['col_names'] = negative_by_label['ticker'] + ' ' + negative_by_label['date_str']
    ''' select these events from CAR '''
    col_names = negative_by_label['col_names'].values.tolist()
    col_names = [col_name for col_name in col_names if col_name in car.columns.tolist()]
    car_sub = car[col_names]

    ''' calcualte mean CAR for these events '''
    mean_car = car_sub.mean(axis=1)
    ''' calculate variance CAR for all events in this category '''
    cols = car_sub.columns.tolist()
    N = len(cols)
    rse = reg_estimation.loc[reg_estimation['rse'].index.isin(cols),'rse']
    var_car_i = (window*2+1) * rse
    std_car = math.sqrt(sum(var_car_i)/N**2)
    t_car = mean_car[window]/std_car
    pval = stats.t.sf(np.abs(t_car), N-1)*2
    
    return mean_car, std_car, pval

mean_dic = {}
pval_dic = {}
std_dic = {}
mean_cum_val = {}
label2 = df_negative['label2'].unique().tolist()
for label in label2:
    print(label)
    mean_car, std_car, pval = car_by_label(label)
    print(mean_car)
    mean_dic[label] = mean_car
    mean_cum_val[label] = mean_car[window]
    pval_dic[label] = round(pval,5)
    std_dic[label] = std_car

''' plot CAR for each event subcategory during the event window'''
fig, axs = plt.subplots(6,3, figsize=(10, 8), constrained_layout=True)
axs = axs.ravel()
for i,var in enumerate(mean_dic):
    x = mean_dic[var].index.values
    y = mean_dic[var].values
    yerr = std_dic[var]
    ymin = y - 1.65*yerr
    ymax = y + 1.65*yerr
    axs[i].plot(x, y)
    axs[i].set_title(var)
plt.show()

''' plot CAR and confidence interval at the end of event window'''
yerror=[std_dic[i]*1.65 for i in std_dic]
mean_window_end = {k: mean_dic[k][window] for k in mean_dic}
plt.figure(figsize=(9,5))
plt.xticks(rotation=90)
x = range(len(mean_window_end))
y = list(mean_window_end.values())
plt.axhline(color='black')
plt.errorbar(x, y, yerr=yerror, fmt='o',lolims=True, uplims=True)
plt.xticks(x, list(mean_window_end.keys()))
plt.title('CARs at the end of window period(day 20) by event category')
plt.xlabel('event category')
plt.ylabel('CAR')
plt.show()
