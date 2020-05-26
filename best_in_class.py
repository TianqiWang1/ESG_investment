import pandas as pd
import numpy as np
import random
from dateutil.relativedelta import relativedelta
from datetime import date, datetime
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

'''read data'''
df_negative = pd.read_pickle('df_negative_new_1.pkl')
df_negative = df_negative[['ticker', 'date','label2']]
df_negative['news_count'] = df_negative.groupby('ticker')['date'].rank(ascending=True)
df_negative = df_negative.sort_values(['ticker', 'date'])


data_ret = pd.read_csv('monthly_return.csv', header = 1)
data_ret = data_ret.iloc[1:,]
data_ret.ticker = data_ret.ticker.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
data_ret.index = data_ret['ticker']
data_ret = data_ret.iloc[:,1:]
data_ret /= 100
data_ret = data_ret.dropna(axis='columns')

stock_tickers = list(set(data_ret.columns) & set(df_negative.ticker.unique().tolist()))
df_negative = df_negative[df_negative.ticker.isin(stock_tickers)]
data_ret = data_ret[stock_tickers]

monthly_ret_all = pd.read_csv('monthly_ret_all_stocks_1.csv')
monthly_ret_all.index = monthly_ret_all['date']
monthly_ret_all = monthly_ret_all.iloc[:,1:]
monthly_ret_all /= 100
monthly_rets_all = monthly_ret_all.mean(axis=1) .values.tolist()

monthly_ret_sp500 = pd.read_csv('monthly_return_sp500.csv', header=1)
monthly_ret_sp500 = monthly_ret_sp500.iloc[1:,:]
monthly_ret_sp500.ticker = monthly_ret_sp500.ticker.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
monthly_ret_sp500.index = monthly_ret_sp500.ticker
monthly_ret_sp500 = monthly_ret_sp500.iloc[:,1:]
monthly_ret_sp500/=100

all_monthly_rets = monthly_ret_all.mean(axis=1).values.tolist()
sp500_monthly_rets = monthly_ret_sp500.mean(axis=1).values.tolist()

event_starting_date = pd.Timestamp(year=2012, month=1, day=1)
event_ending_date = relativedelta(months=1) + event_starting_date
portfolio_starting_date = event_ending_date
portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)
df = df_negative[(df_negative.date < event_ending_date)&(df_negative.date >= event_starting_date)]

'''value-weighted portfolio returns for worst 30 stocks'''

event_starting_date = pd.Timestamp(year=2012, month=1, day=1)
event_ending_date = pd.Timestamp(year=2012, month=4, day=1)
df = df_negative[(df_negative.date < event_ending_date)&(df_negative.date >= event_starting_date)]
tickers = ['a' for i in range(30)]
monthly_rets = []
turn_overs = []

while event_ending_date < pd.Timestamp(year=2019, month=1,day=1):
    
    portfolio_starting_date = event_ending_date
    portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)

    df = df_negative[(df_negative.date < event_ending_date)]
    news_count = pd.DataFrame(df.groupby('ticker')['news_count'].max())
    prev_tickers = tickers
    tickers = news_count.sort_values('news_count', ascending=False)[:30].index.tolist()
    turnover_rate = 1-len(list(set(prev_tickers) & set(tickers)))/30
    turn_overs.append(turnover_rate)
    rets = data_ret[tickers][portfolio_starting_date:portfolio_ending_date].values[0]
    mc = mc_all[tickers][portfolio_starting_date:portfolio_ending_date]
    mc_sum = np.sum(mc,axis=1).values[0]
    monthly_ret = np.sum((rets * mc)/ mc_sum ,axis=1).values[0]
    monthly_rets.append(monthly_ret)
    event_ending_date += relativedelta(months=1)

monthly_rets_worst = monthly_rets

''' value-weighted portfolio returns for best 30 stocks '''
event_starting_date = pd.Timestamp(year=2012, month=1, day=1)
event_ending_date = pd.Timestamp(year=2012, month=4, day=1)
df = df_negative[(df_negative.date < event_ending_date)&(df_negative.date >= event_starting_date)]
tickers = ['a' for i in range(30)]
monthly_rets = []
turn_overs = []

while event_ending_date < pd.Timestamp(year=2019, month=1,day=1):
    
    portfolio_starting_date = event_ending_date
    portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)

    df = df_negative[(df_negative.date < event_ending_date)]
    news_count = pd.DataFrame(df.groupby('ticker')['news_count'].max())
    prev_tickers = tickers
    tickers = news_count.sort_values('news_count', ascending=True)[:30].index.tolist()
    turnover_rate = 1-len(list(set(prev_tickers) & set(tickers)))/30
    turn_overs.append(turnover_rate)
    rets = data_ret[tickers][portfolio_starting_date:portfolio_ending_date].values[0]
    mc = mc_all[tickers][portfolio_starting_date:portfolio_ending_date]
    mc_sum = np.sum(mc,axis=1).values[0]
    monthly_ret = np.sum((rets * mc)/ mc_sum ,axis=1).values[0]
    monthly_rets.append(monthly_ret)
    event_ending_date += relativedelta(months=1)

'''value-weighted benchmark returns'''
mc_sp500 = pd.read_csv('mc_sp500.csv',header = 1)
mc_sp500 = mc_sp500.iloc[1:,:]
mc_sp500.ticker = mc_sp500.ticker.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
mc_sp500.index = mc_sp500.ticker
mc_sp500 = mc_sp500.iloc[:,1:]
mc_all = pd.read_csv('monthly_market_cap_all_stocks_1.csv')
mc_all.date = mc_all.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
mc_all.index = mc_all.date
mc_all = mc_all.iloc[:,1:]
all_tickers =  mc_all.columns
all_tickers = [t.split('.', 1)[0].lower() for t in all_tickers]
mc_all.columns = all_tickers
all_tickers =  monthly_ret_all.columns
all_tickers = [t.split('.', 1)[0].lower() for t in all_tickers]
monthly_ret_all.columns = all_tickers
stocks_sp500 = list(set(mc_sp500.columns) & set(monthly_ret_sp500.columns))
mc_sp500 = mc_sp500[stocks_sp500]
monthly_ret_sp500 = monthly_ret_sp500[stocks_sp500]
stocks_all = list(set(mc_all.columns) & set(monthly_ret_all.columns))
mc_all = mc_all[stocks_all]
monthly_ret_all = monthly_ret_all[stocks_all]
sp500_weighted_returns = (monthly_ret_sp500 * mc_sp500).div(mc_sp500.sum(axis=1),axis=0).sum(axis=1).values.tolist()
all_weighted_return = (monthly_ret_all * mc_all).div(mc_all.sum(axis=1),axis=0).sum(axis=1).values.tolist()
tickers = list(set(df_negative.ticker.unique()) & set(mc_all.columns))
df_negative = df_negative[df_negative.ticker.isin(tickers)]

'''compare portfolio returns with the benchmark'''
cumsum_all_stocks = np.cumsum(all_monthly_rets[5:-1])
cumsum_sp500 = np.cumsum(sp500_monthly_rets[5:-1])
cumsum_worst = np.cumsum(monthly_rets_worst)
cumsum_best = np.cumsum(monthly_rets_best)
date = pd.date_range(start='2012-04-01', end='2018-12-31', freq='MS')
data = pd.DataFrame(data = {'date': date,'sp500': cumsum_sp500, 'all public': cumsum_all_stocks,
                    'best 30': cumsum_best, 'worst 30': cumsum_worst})

plt.plot('date', 'sp500', data = data)
plt.plot('date', 'all public', data = data)
plt.plot('date', 'best 30', data = data)
plt.plot('date', 'worst 30', data = data)
plt.xlabel('year')
plt.ylabel('cumulative return')
plt.legend()

factors = pd.read_pickle('factors_monthly.pkl')
factors = factors[(factors.index >= '2012-4-1') & (factors.index < '2019-1-1')]

''' Carhart four-factor model '''
factors['R'] = monthly_rets
factors['RF'] = all_monthly_rets[5:-1]
factors['R-RF'] = factors['R'] -factors['RF']
factors['const'] = 1
factors.columns = ['Mkt-RF', 'SMB', 'HML', 'RF', 'MKT', 'MOM', 'R', 'R-RF', 'const']
mod = sm.OLS(factors['R-RF'], factors[['MKT', 'HML', 'SMB', 'MOM','const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()