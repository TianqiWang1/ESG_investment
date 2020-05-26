import pandas as pd
import numpy as np
import random
from dateutil.relativedelta import relativedelta
from datetime import date, datetime
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

''' read  data '''
df_negative = pd.read_pickle('df_negative_new_1.pkl')
df_negative = df_negative[['ticker', 'date','label2','label1']]
df_negative['news_count'] = df_negative.groupby('ticker')['date'].rank(ascending=True)
df_negative = df_negative.sort_values(['ticker', 'date'])

monthly_ret_all = pd.read_csv('monthly_ret_all_stocks_1.csv')
monthly_ret_all.date = monthly_ret_all.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
monthly_ret_all.index = monthly_ret_all['date']
monthly_ret_all = monthly_ret_all.iloc[:,1:]
monthly_ret_all /= 100

all_tickers =  monthly_ret_all.columns
all_tickers = [t.split('.', 1)[0].lower() for t in all_tickers]
monthly_ret_all.columns = all_tickers

monthly_ret_sp500 = pd.read_csv('monthly_return_sp500.csv', header=1)
monthly_ret_sp500 = monthly_ret_sp500.iloc[1:,:]
monthly_ret_sp500.ticker = monthly_ret_sp500.ticker.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
monthly_ret_sp500.index = monthly_ret_sp500.ticker
monthly_ret_sp500 = monthly_ret_sp500.iloc[:,1:]
monthly_ret_sp500/=100

all_monthly_rets = monthly_ret_all.mean(axis=1).values.tolist()
sp500_monthly_rets = monthly_ret_sp500.mean(axis=1).values.tolist()

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
stocks_sp500 = list(set(mc_sp500.columns) & set(monthly_ret_sp500.columns))
mc_sp500 = mc_sp500[stocks_sp500]
monthly_ret_sp500 = monthly_ret_sp500[stocks_sp500]

stocks_all = list(set(mc_all.columns) & set(monthly_ret_all.columns))
mc_all = mc_all[stocks_all]
monthly_ret_all = monthly_ret_all[stocks_all]
sp500_weighted_returns = (monthly_ret_sp500 * mc_sp500).div(mc_sp500.sum(axis=1),axis=0).sum(axis=1).values.tolist()
all_weighted_return = (monthly_ret_all * mc_all).div(mc_all.sum(axis=1),axis=0).sum(axis=1).values.tolist()


'''value-weighted negatively screening portfolios'''

'''perform negative screening on all public traded US stocks'''
threshold = 0
event_ending_date = pd.Timestamp(year=2012, month=2, day=1)
monthly_rets_all_public = []
num_stocks_list = []
stock_tickers = monthly_ret_all.columns.tolist()

while event_ending_date < pd.Timestamp(year=2019, month=1,day=1):
    
    portfolio_starting_date = event_ending_date
    portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)
    df = df_negative[(df_negative.date < event_ending_date)]
    count = pd.DataFrame(df.groupby('ticker')['news_count'].max())
    negative_tickers = count[count.news_count > threshold].index.tolist()
    tickers = [x for x in stock_tickers if x not in negative_tickers]
    num_stocks = len(tickers)
    num_stocks_list.append(num_stocks)
    rets = monthly_ret_all[tickers][portfolio_starting_date:portfolio_ending_date]
    mc = mc_all[tickers][portfolio_starting_date:portfolio_ending_date]
    mc_sum = np.sum(mc,axis=1).values[0]
    monthly_ret = np.sum((rets * mc)/ mc_sum ,axis=1).values[0]
    monthly_rets_all_public.append(monthly_ret)
    event_ending_date += relativedelta(months=1)

factors['R'] = monthly_rets_all_public
factors['RF'] = all_weighted_return[3:-1]
factors['R-RF'] = factors['R'] - factors['RF']
mod = sm.OLS(factors['R-RF'], factors[['SMB', 'HML', 'MKT', 'MOM', 'const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()

'''perform negative screening on SP500 stocks'''

threshold = 0
event_ending_date = pd.Timestamp(year=2012, month=2, day=1)
monthly_rets_sp500 = []
num_stocks_list = []
stock_tickers = monthly_ret_sp500.columns.tolist()

while event_ending_date < pd.Timestamp(year=2019, month=1,day=1):   
    portfolio_starting_date = event_ending_date
    portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)
    df = df_negative[(df_negative.date < event_ending_date)]
    count = pd.DataFrame(df.groupby('ticker')['news_count'].max())
    negative_tickers = count[count.news_count > threshold].index.tolist()
    tickers = [x for x in stock_tickers if x not in negative_tickers]
    num_stocks = len(tickers)
    num_stocks_list.append(num_stocks)
    rets = monthly_ret_sp500[tickers][portfolio_starting_date:portfolio_ending_date]
    mc = mc_sp500[tickers][portfolio_starting_date:portfolio_ending_date]
    mc_sum = np.sum(mc,axis=1).values[0]
    monthly_ret = np.sum((rets * mc)/ mc_sum ,axis=1).values[0]
    monthly_rets_sp500.append(monthly_ret)
    event_ending_date += relativedelta(months=1)

factors['R'] = monthly_rets_sp500
factors['RF'] = sp500_weighted_returns[3:-1]
factors['R-RF'] = factors['R'] - factors['RF']
mod = sm.OLS(factors['R-RF'], factors[['SMB', 'HML', 'MKT', 'MOM', 'const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()

'''plot cumulative returns'''

date = pd.date_range(start='2012-02-01', end='2018-12-31', freq='MS')
cumsum_all_stocks = np.cumsum(all_weighted_return[3:-1])
cumsum_sp500 = np.cumsum(sp500_weighted_returns[3:-1])
cumsum_all_screened = np.cumsum(monthly_rets_all_public)
cumsum_sp500_screened = np.cumsum(monthly_rets_sp500)
data = pd.DataFrame(data = {'date': date,'sp500': cumsum_sp500, 'all public': cumsum_all_stocks,
                    'all public negative screened': cumsum_all_screened , 
                    'sp500 negative screened': cumsum_sp500_screened})
plt.plot('date', 'sp500', data = data)
plt.plot('date', 'sp500 negative screened', data = data)
plt.xlabel('year')
plt.ylabel('cumulative return')
plt.legend()
