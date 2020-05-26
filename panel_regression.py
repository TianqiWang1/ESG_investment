import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import warnings
import datetime as dt
from scipy import stats
import pandas_datareader.data as web
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
from linearmodels import PooledOLS

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

''' preprocess'''

''' dividend yield '''
div = pd.read_csv('div_yield.csv', header = 1)
div = div.iloc[1:,:]
div.index = div.ticker
div = div.iloc[:,1:]

''' log of market capitalization, in terms of billion of dollar '''
mc = pd.read_csv('market_cap.csv', header = 1)
mc = mc.iloc[1:,:]
mc.index = mc.ticker
mc = mc.iloc[:,1:]
mc /= 1000000000
mc = np.log(mc)

''' log of book-to-market ratio '''
bm = pd.read_csv('book_to_market_ratio.csv', header=1)
bm = bm.iloc[1:,:]
bm.index = bm.ticker
bm = bm.iloc[:,1:]
bm = np.log(bm)

''' log of reciprocal of stock price '''
price = pd.read_csv('monthly_price.csv', header=1)
price = price.iloc[1:,:]
price.index = price.ticker
price = price.iloc[:,1:]
price = np.log(price)

prc = 1/price
prc = np.log(prc)

''' log of cumulative returns '''
ret23 = np.log((price.shift(2)/price.shift(3)))
ret46 = np.log((price.shift(4)/price.shift(6)))
ret712 = np.log((price.shift(7)/price.shift(12)))

div = div.shift(1)
mc = mc.shift(1)
bm = bm.shift(1)
prc = prc.shift(1)

df_negative = pd.read_pickle('df_negative_new_1.pkl')
df_negative = df_negative[['ticker', 'date', 'label1', 'label2']]
df_negative.date = df_negative.date + pd.offsets.MonthEnd(0)
df_negative.index = df_negative.ticker
df_negative = df_negative[['date', 'label1', 'label2']]
df_negative_1 = df_negative

''' create a dummy, which is the count of negative events for the last month'''
df_negative = pd.get_dummies(df_negative['date']).groupby(df_negative.index).sum()
df_negative.columns = [i + relativedelta(months=1) + pd.offsets.MonthEnd(0) for i in df_negative.columns]
df_negative['tickers'] = df_negative.index
df_negative = pd.melt(df_negative, 
        id_vars= 'tickers', 
        value_vars=list(df_negative.columns)[:-1], 
        var_name='date', 
        value_name='dummy')
df_negative.date = df_negative.date.apply(lambda x: x.strftime('%Y-%m-%d'))

''' create a set of dummies, which count the negative events for the last month per event subcategory '''
labels = df_negative_1.label2.unique().tolist()
l = labels[0]
df_negative = df_negative_1[df_negative_1.label2 == l]
df_negative = df_negative[['date']]
df_negative = pd.get_dummies(df_negative['date']).groupby(df_negative.index).sum()
df_negative.columns = [i + relativedelta(months=1) + pd.offsets.MonthEnd(0)  for i in df_negative.columns]
df_negative['tickers'] = df_negative.index
df_negative = pd.melt(df_negative, 
        id_vars= 'tickers', 
        value_vars=list(df_negative.columns)[:-1], 
        var_name='date', 
        value_name='dummy')
df_negative.date = df_negative.date.apply(lambda x: x.strftime('%Y-%m-%d'))
df_negative.columns = ['tickers', 'date', l]

for l in labels[1:]:
    df_new = df_negative_1[df_negative_1.label2 == l]
    df_new = df_new[['date']]
    df_new = pd.get_dummies(df_new['date']).groupby(df_new.index).sum()
    df_new.columns = [i + relativedelta(months=1) + pd.offsets.MonthEnd(0)  for i in df_new.columns]
    df_new['tickers'] = df_new.index
    df_new = pd.melt(df_new, 
        id_vars= 'tickers', 
        value_vars=list(df_new.columns)[:-1], 
        var_name='date', 
        value_name='dummy')
    df_new.date = df_new.date.apply(lambda x: x.strftime('%Y-%m-%d'))
    df_new.columns = df_new.columns[:-1].tolist() + [l]
    df_negative = df_new.merge(df_negative, on = ['tickers', 'date'], how = 'outer')
df_negative = df_negative.fillna(0)

mc = mc.iloc[2:,:]
mc = mc.iloc[:,(mc.values < np.percentile(mc,99)).all(axis=0) & (mc.values > np.percentile(mc,1)).all(axis=0)]

prc = prc.iloc[2:,:]
prc = prc.dropna(axis=1)
prc = prc.iloc[:,(prc.values < np.percentile(prc,99)).all(axis=0) & (prc.values > np.percentile(prc,1)).all(axis=0)]

div = div.iloc[2:,:]
div = div.dropna(axis=1)
div = div.iloc[:,(div.values < np.percentile(div,99)).all(axis=0) & (div.values > np.percentile(div,1)).all(axis=0)]

bm = bm.iloc[2:,:]
bm = bm.dropna(axis=1)
bm = bm.iloc[:,(bm.values < np.percentile(bm,99)).all(axis=0) & (bm.values > np.percentile(bm,1)).all(axis=0)]

ret23 = ret23.iloc[3:,:]
ret23 = ret23.dropna(axis=1)
ret23 = ret23.iloc[:,(ret23.values < np.percentile(ret23_1,99)).all(axis=0) & (ret23.values > np.percentile(ret23_1,1)).all(axis=0)]

ret46 = ret46.iloc[6:,:]
ret46 = ret46.dropna(axis=1)
ret46 = ret46.iloc[:,(ret46.values < np.percentile(ret46_1,99)).all(axis=0) & (ret46.values > np.percentile(ret46_1,1)).all(axis=0)]

ret712 = ret712.iloc[12:,:]
ret712 = ret712.dropna(axis=1)
ret712 = ret712.iloc[:,(ret712.values < np.percentile(ret712_1,99)).all(axis=0) & (ret712.values > np.percentile(ret712_1,1)).all(axis=0)]

monthly_return = pd.read_csv('monthly_return.csv',header=1)
monthly_return = monthly_return.iloc[1:,:]
monthly_return.index = monthly_return.ticker

monthly_return= monthly_return.drop(columns = ['ticker'])
monthly_return = monthly_return.dropna(axis=1)
mc_max_cut = np.percentile(monthly_return,99)
mc_min_cut = np.percentile(monthly_return,1)
monthly_return = monthly_return.iloc[:,(monthly_return.values < mc_max_cut).all(axis=0) & 
                           (monthly_return.values > mc_min_cut).all(axis=0)]

monthly_return['ticker'] = monthly_return.index
monthly_return = pd.melt(monthly_return, 
        id_vars= 'ticker', 
        value_vars=list(monthly_return.columns)[:-1], 
        var_name='tickers', 
        value_name='return')
monthly_return.columns = ['date', 'tickers', 'return']
monthly_return['return'] = monthly_return['return']/100

data = df_negative.merge(mc, how = 'inner', on = ['tickers', 'date'])\
.merge(prc, how = 'inner', on = ['tickers', 'date'])\
.merge(bm, how = 'inner', on = ['tickers', 'date']).merge(div, how = 'inner', on = ['tickers', 'date'])\
.merge(ret23, how = 'inner', on = ['tickers', 'date']).merge(ret46, how = 'inner', on = ['tickers', 'date'])\
.merge(ret712, how = 'inner', on = ['tickers', 'date']).merge(monthly_return, how='inner', on=['tickers','date'])
data.columns = df_negative.columns.tolist() + ['mc', 'prc', 'bm', 'div', 'ret23', 'ret46','ret712', 'ret']

data = data.replace([np.inf, -np.inf], np.nan)
data1 = data.dropna()

#data1['dummy'] = data1['dummy'].clip(0, 1)

'''Fama-Macbeth regression'''
def ols_coef(x,formula):
    return smf.ols(formula,data=x).fit().params

def fm_summary(p):
    s = p.describe().T
    s['std_error'] = s['std']/np.sqrt(s['count'])
    s['tstat'] = s['mean']/s['std_error']
    s['pval'] = stats.t.sf(np.abs(s['tstat']), s['count']-1)*2
    return s[['mean','std_error','tstat', 'pval']]

gamma = data1.groupby('date').apply(ols_coef,'ret ~ 1 + dummy + mc + prc + bm + div + ret23 + ret46 + ret712')
res = fm_summary(gamma)
res.pval = [round(x,3) for x in res.pval.values.tolist()]

gamma = data1.groupby('date').apply(ols_coef,'ret ~ 1 +Association+Sanctions+Financial+Corruption+Information+Human+Workplace+Production_Supply+Environmental+Management+Workforce+Regulatory+Fraud+Anti_Competitive+Ownership+Product_Service+Discrimination_Workforce+mc + prc + bm + div + ret23 + ret46 + ret712')
res = fm_summary(gamma)
res.pval = [round(x,3) for x in res.pval.values.tolist()]
res

'''pooled OLS with double-clustered standard error'''
data1.date = data1.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
data1 = data1.set_index(['tickers','date'])
data1['const'] = 1
mod = PooledOLS(data1['ret'], data1[['dummy','mc','prc','bm','div','ret23','ret46','ret712', 'const']])
res = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time = True)
