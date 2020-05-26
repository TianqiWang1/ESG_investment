import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import datetime as dt
from scipy import stats
import math
import operator
from statsmodels.graphics.tsaplots import plot_acf

''' read and preprocess data'''
factors = pd.read_pickle('factors_monthly.pkl')
factors = factors[(factors.index >= '2013-1-1') & (factors.index < '2019-2-1')]

factors_5 = pd.read_csv('fama_french_5_factor.csv')
factors_5 = factors_5.iloc[12:-5]
factors_5 = factors_5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']]
factors_5['const'] = 1

data_ret = pd.read_csv('monthly_return.csv', header=1)
data_ret = data_ret.iloc[1:,:]
data_ret.index = data_ret.ticker
data_ret = data_ret.iloc[:,1:]
data_ret /= 100
data_ret['date'] = data_ret.index
data_ret.index = data_ret.date.apply(lambda x: datetime.strptime(x,'%Y-%m-%d')) 
data_ret = data_ret.drop('date', axis=1)
data_ret = data_ret.dropna(axis=1)
data_ret_1 = data_ret.dropna(axis=1)
data_ret= data_ret.iloc[:,(data_ret.values < np.percentile(data_ret_1,99)).all(axis=0) & (data_ret.values > np.percentile(data_ret_1,1)).all(axis=0)]

df_negative = pd.read_pickle('df_negative_new_1.pkl')
stock_tickers = data_ret.columns.tolist()

''' equal-weighted portfolio'''
event_starting_date = pd.Timestamp(year=2012, month=1, day=1)
monthly_rets = []
num_firms = []
while event_starting_date < pd.Timestamp(year=2018, month=2,day=1):
    event_ending_date = relativedelta(months=12) + event_starting_date
    portfolio_starting_date = event_ending_date
    portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)

    ''' find stocks with negative news within the last 12 months '''
    df = df_negative[(df_negative.date < event_ending_date)&(df_negative.date >= event_starting_date)]
    tickers = df.ticker.unique().tolist()
    new_tickers = list(set(tickers) & set(stock_tickers))
    num_firms.append(len(new_tickers))

    ''' compute monthly return for every stock '''
    rets = data_ret[new_tickers][portfolio_starting_date:portfolio_ending_date]
    monthly_ret = np.mean(rets, axis=1).values[0]
    monthly_rets.append(monthly_ret)
    event_starting_date += relativedelta(months=1)

'''Fama-French five-factor model'''
factors_5['R'] = monthly_rets
factors_5['R-RF'] = factors_5['R'] - factors_5['RF']
mod = sm.OLS(factors_5['R-RF'], factors_5[['SMB', 'HML', 'Mkt-RF','RMW', 'CMA', 'const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()

'''Carhart four-factor model'''
factors['R'] = monthly_rets
factors['R-RF'] = factors['R'] -factors['RF']
factors['const'] = 1
factors.columns = ['Mkt-RF', 'SMB', 'HML', 'RF', 'MKT', 'Mom', 'R', 'R-RF', 'const']
mod = sm.OLS(factors['R-RF'], factors[['SMB', 'HML', 'MKT','Mom', 'const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()

'''Fama-French three-factor model'''
mod = sm.OLS(factors['R-RF'], factors[['SMB', 'HML', 'Mkt-RF', 'const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()

'''residual and ACF plot'''
resids = res.resid.values.tolist()
y_fitted = res.fittedvalues
plot_acf(resids, lags=20)

fig, ax = plt.subplots(figsize=(6,3))
_ = ax.scatter(y_fitted, resids)
plt.xlabel('fitted values')
plt.ylabel('residual')
plt.title('residual vs. fitted value')

''' run the regression by event category'''
labels = df_negative.label2.unique().tolist()
alphas = []
pvals = []
num_firms_list = []

for label in labels:
    df_sub = df_negative[df_negative.label2 == label]
    print('==== currently running regression for label: {}'.format(label))
    
    event_starting_date = pd.Timestamp(year=2012, month=1, day=1)
    monthly_rets = []
    num_firms = []

    while event_starting_date < pd.Timestamp(year=2018, month=2,day=1):
        event_ending_date = relativedelta(months=12) + event_starting_date
        portfolio_starting_date = event_ending_date
        portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)

        df = df_sub[(df_sub.date < event_ending_date)&(df_sub.date >= event_starting_date)]
        tickers = df.ticker.unique().tolist()
        new_tickers = list(set(tickers) & set(stock_tickers))
        num_firms.append(len(new_tickers))

        rets = data_ret[new_tickers][portfolio_starting_date:portfolio_ending_date]
        monthly_ret = np.mean(rets, axis=1).values[0]
        monthly_rets.append(monthly_ret)
        event_starting_date += relativedelta(months=1)
    factors['R'] = monthly_rets
    factors['R-RF'] = factors['R'] -factors['RF']
    mod = sm.OLS(factors['R-RF'], factors[['SMB', 'HML', 'MKT', 'Mom', 'const']])
    res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    
    alphas.append(res.params[-1])
    pvals.append(res.pvalues[-1])
    num_firms_list.append(np.mean(num_firms))

d = {'label': labels, 'alpha': alphas, 'p-value': pvals, 'num_firms': num_firms_list}
res = pd.DataFrame(d)
res.index = res.label
res = res.drop(columns = 'label')
res['num_firms'] = res['num_firms'].astype(int)
res = res.sort_values('num_firms', ascending=False)
res = res[['alpha', 'p-value', 'num_firms']]
res

'''value-weighted portfolios'''
market_cap = pd.read_csv('market_cap.csv', header=1)
market_cap = market_cap.iloc[1:,:]
market_cap.index = market_cap.ticker.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
market_cap = market_cap.drop('ticker', axis=1)
market_cap_1 = market_cap.dropna(axis=1)
mc_max_cut = np.percentile(market_cap_1,99)
mc_min_cut = np.percentile(market_cap_1,1)
market_cap_1 = market_cap.iloc[:,(market_cap.values < mc_max_cut).all(axis=0) & (market_cap.values > mc_min_cut).all(axis=0)]
market_cap = market_cap_1

event_starting_date = pd.Timestamp(year=2012, month=1, day=1)
monthly_rets = []
num_firms = []
while event_starting_date < pd.Timestamp(year=2018, month=2,day=1):
    event_ending_date = relativedelta(months=12) + event_starting_date
    portfolio_starting_date = event_ending_date
    portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)

    df = df_negative[(df_negative.date < event_ending_date)&(df_negative.date >= event_starting_date)]
    tickers = df.ticker.unique().tolist()
    new_tickers = list(set(tickers) & set(stock_tickers) & set(market_cap.columns))
    num_firms.append(len(new_tickers))
    rets = data_ret[new_tickers][portfolio_starting_date:portfolio_ending_date]
    mc = market_cap[new_tickers][portfolio_starting_date:portfolio_ending_date]
    rets = rets[sorted(rets.columns)]
    mc = rets[sorted(mc.columns)]
    if (sorted(mc.columns) != sorted(rets.columns)) or (len(mc.columns) != len(rets.columns)):
        print('False!')
    value_weighted_rets = mc*rets/np.sum(mc, axis=1).values[0]
    monthly_ret = np.mean(value_weighted_rets, axis=1).values[0]
    
    monthly_rets.append(monthly_ret)
    event_starting_date += relativedelta(months=1)

'''value-weighted four-factor model'''
factors['R'] = monthly_rets
factors['R-RF'] = factors['R'] -factors['RF']
mod = sm.OLS(factors['R-RF'], factors[['SMB', 'HML', 'MKT','Mom', 'const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()

'''winsorize stock returns'''
q95 = np.percentile(data_ret.values,95)
q5 = np.percentile(data_ret.values,5)
data_ret_1 = data_ret.iloc[:,(data_ret.values < q95).all(axis=0) & (data_ret.values > q5).all(axis=0)]

stock_tickers = data_ret_1.columns.tolist()
event_starting_date = pd.Timestamp(year=2012, month=1, day=1)
monthly_rets = []
num_firms = []
while event_starting_date < pd.Timestamp(year=2018, month=2,day=1):
    event_ending_date = relativedelta(months=12) + event_starting_date
    portfolio_starting_date = event_ending_date
    portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)

    df = df_negative[(df_negative.date < event_ending_date)&(df_negative.date >= event_starting_date)]
    tickers = df.ticker.unique().tolist()
    new_tickers = list(set(tickers) & set(stock_tickers))
    num_firms.append(len(new_tickers))

    rets = data_ret_1[new_tickers][portfolio_starting_date:portfolio_ending_date]
    monthly_ret = np.mean(rets, axis=1).values[0]
    monthly_rets.append(monthly_ret)
    event_starting_date += relativedelta(months=1)

factors['R'] = monthly_rets
factors['R-RF'] = factors['R'] -factors['RF']
factors['const'] = 1
factors.columns = ['Mkt-RF', 'SMB', 'HML', 'RF', 'MKT', 'Mom', 'R', 'R-RF', 'const']
mod = sm.OLS(factors['R-RF'], factors[['SMB', 'HML', 'MKT','Mom', 'const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()

'''industry-matched portfolio'''

industry = pd.read_csv('industry.csv')
industry = industry[['ticker', 'industry', 'industry_sub']]
industry.columns = ['tiker', 'industry1', 'industry2']
industry.loc[(industry.industry1 == 'Food, Beverage & Tobacco') |
             (industry.industry1 == 'Household & Personal Products') |
             (industry.industry2 == 'Textiles, Apparel & Luxury Goods')
             , 'industry'] = 'NoDur'

industry.loc[(industry.industry2 == 'Household Durables') |
             (industry.industry2 == 'Leisure Products')
             , 'industry'] = 'Durbl'

industry.loc[(industry.industry1 == 'Automobiles & Components') |
             (industry.industry1 == 'Capital Goods') |
             (industry.industry2 == 'Paper & Forest Products')
             , 'industry'] = 'Manuf'

industry.loc[(industry.industry1 == 'Software & Services') |
             (industry.industry1 == 'Technology Hardware & Equipment') |
             (industry.industry1 == 'Semiconductors & Semiconductor Equipment')
             , 'industry'] = 'BusEq'

industry.loc[industry.industry1 == 'Energy', 'industry'] = 'Enrgy'
industry.loc[industry.industry1 == 'Telecommunication Services', 'industry'] = 'Telcm'
industry.loc[industry.industry1 == 'Utilities', 'industry'] = 'Utils'
industry.loc[industry.industry2 == 'Chemicals', 'industry'] = 'Chems'

industry.loc[(industry.industry1 == 'Retailing') |
             (industry.industry1 == 'Food & Staples Retailing') , 'industry'] ='Shops'

industry.loc[(industry.industry1 == 'Health Care Equipment & Services') |
             (industry.industry1 == 'Pharmaceuticals, Biotechnology & Life Sciences') 
             , 'industry'] ='Hlth'

industry.loc[(industry.industry1 == 'Diversified Financials') |
             (industry.industry1 == 'Insurance') |
             (industry.industry1 == 'Banks') |
              (industry.industry1 == 'Real Estate')
             , 'industry'] ='Money'

industry.loc[(industry.industry1 == 'Media & Entertainment') |
             (industry.industry1 == 'Consumer Services') |
             (industry.industry1 == 'Commercial  & Professional Services') |
            (industry.industry1 == 'Transportation') |
             (industry.industry2 == 'Metals & Mining') |
             (industry.industry2 == 'Construction Materials') |
             (industry.industry2 == 'Containers & Packaging')
             , 'industry'] ='Other'

industry = industry.dropna(subset = ['industry'])

'''substract the industry return from the company monthly return'''
industry = industry.drop(columns = ['Unnamed: 0'])
industry_portfolios = pd.read_csv('12_portfolio_equal_weighted.csv')
industry_portfolios['Unnamed: 0'] = industry_portfolios['Unnamed: 0'].apply(lambda x: datetime.strptime(str(x),'%Y%m'))
industry_portfolios.columns = ['date', 'NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems', 'BusEq',
       'Telcm', 'Utils', 'Shops', 'Hlth ', 'Money', 'Other']
industry_portfolios.index = industry_portfolios.date
industry_portfolios = industry_portfolios.iloc[:,1:]
industry_portfolios/=100
industry_portfolios['date'] = industry_portfolios.index
industry_portfolios = industry_portfolios[(industry_portfolios.date >= datetime.strptime('2011-11-01','%Y-%m-%d'))
                                          & (industry_portfolios.date <= datetime.strptime('2019-01-01','%Y-%m-%d'))]
data_ret_1 = data_ret
data_ret_1.index = data_ret_1.index - pd.tseries.offsets.MonthEnd() + timedelta(days=1)

industry_portfolios.columns = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems', 'BusEq', 'Telcm', 'Utils',
       'Shops', 'Hlth', 'Money', 'Other', 'date']

for t in data_ret_1.columns:
    indstr = industry[industry.ticker == t].industry.values.tolist()
    if len(indstr) == 0:
        data_ret_1.drop(columns = t,inplace=True)
    else:
        indstr = indstr[0]
        data_ret_1[t] -= industry_portfolios[indstr]

''' construct the industry-matched portfolio'''
stock_tickers = data_ret_1.columns.tolist()
event_starting_date = pd.Timestamp(year=2012, month=1, day=1)
monthly_rets = []
num_firms = []
while event_starting_date < pd.Timestamp(year=2018, month=2,day=1):
    event_ending_date = relativedelta(months=12) + event_starting_date
    portfolio_starting_date = event_ending_date
    portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)
    df = df_negative[(df_negative.date < event_ending_date)&(df_negative.date >= event_starting_date)]
    tickers = df.ticker.unique().tolist()
    new_tickers = list(set(tickers) & set(stock_tickers))
    num_firms.append(len(new_tickers))
    rets = data_ret_1[new_tickers][portfolio_starting_date:portfolio_ending_date]
    monthly_ret = np.mean(rets, axis=1).values[0]
    monthly_rets.append(monthly_ret)
    event_starting_date += relativedelta(months=1)
    
factors['R'] = monthly_rets
factors['R-RF'] = factors['R'] -factors['RF']
factors['const'] = 1
factors.columns = ['Mkt-RF', 'SMB', 'HML', 'RF', 'MKT', 'Mom', 'R', 'R-RF', 'const']
mod = sm.OLS(factors['R'], factors[['SMB', 'HML', 'MKT','Mom', 'const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()

'''characteristics matched portfolio '''
mc_all_stocks = pd.read_csv('market_cap_all_stocks_1.csv')
mc_all_stocks.date = mc_all_stocks.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
mc_all_stocks.index = mc_all_stocks.date
mc_all_stocks = mc_all_stocks.drop(columns= ['date'])

bv_all_stocks = pd.read_csv('bv_all_stocks_1.csv')
bv_all_stocks.date = bv_all_stocks.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
bv_all_stocks.index = bv_all_stocks.date
bv_all_stocks = bv_all_stocks.drop(columns= ['date'])

yearly_ret_all_stocks = pd.read_csv('yearly_return_all_stocks_1.csv')
yearly_ret_all_stocks.date = yearly_ret_all_stocks.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
yearly_ret_all_stocks.index = yearly_ret_all_stocks.date
yearly_ret_all_stocks = yearly_ret_all_stocks.drop(columns= ['date'])

mc_all_stocks = mc_all_stocks.dropna(axis=1)
bv_all_stocks = bv_all_stocks.dropna(axis=1)
yearly_ret_all_stocks = yearly_ret_all_stocks.dropna(axis=1)

mc_esg_stocks = pd.read_csv('yearly_market_cap_esg_stocks.csv',header=1)
mc_esg_stocks = mc_esg_stocks.iloc[1:,:]
mc_esg_stocks.index = mc_esg_stocks.ticker
mc_esg_stocks = mc_esg_stocks.iloc[:,1:]

bv_esg_stocks = pd.read_csv('bv_esg_stocks.csv',header=1)
bv_esg_stocks = bv_esg_stocks.iloc[1:,:]
bv_esg_stocks.index = bv_esg_stocks.ticker
bv_esg_stocks = bv_esg_stocks.iloc[:,1:]

yearly_ret_esg_stocks = pd.read_csv('yearly_return_esg_stocks.csv',header=1)
yearly_ret_esg_stocks = yearly_ret_esg_stocks.iloc[1:,:]
yearly_ret_esg_stocks.index = yearly_ret_esg_stocks.ticker
yearly_ret_esg_stocks = yearly_ret_esg_stocks.iloc[:,1:]

mc_esg_stocks.columns =[t.split('.', 1)[0].lower() for t in mc_esg_stocks.columns.tolist()]
bv_esg_stocks.columns =[t.split('.', 1)[0].lower() for t in bv_esg_stocks.columns.tolist()]
yearly_ret_esg_stocks.columns =[t.split('.', 1)[0].lower() for t in yearly_ret_esg_stocks.columns.tolist()]
mc_all_stocks.columns =[t.split('.', 1)[0].lower() for t in mc_all_stocks.columns.tolist()]
bv_all_stocks.columns =[t.split('.', 1)[0].lower() for t in bv_all_stocks.columns.tolist()]
yearly_ret_all_stocks.columns = [t.split('.', 1)[0].lower() for t in yearly_ret_all_stocks.columns.tolist()]

monthly_ret_all_stocks = pd.read_csv('monthly_ret_all_stocks_1.csv')
monthly_ret_all_stocks.date = monthly_ret_all_stocks.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
monthly_ret_all_stocks.index = monthly_ret_all_stocks['date']
monthly_ret_all_stocks = monthly_ret_all_stocks.iloc[:,1:]
monthly_ret_all_stocks /= 100
monthly_ret_all_stocks.columns = [t.split('.', 1)[0].lower() for t in monthly_ret_all_stocks.columns.tolist()]
d = [mc_all_stocks.columns, bv_all_stocks.columns, yearly_ret_all_stocks.columns, monthly_ret_all_stocks.columns]
tickers = list(set.intersection(*map(set,d)))
tickers = [t for t in tickers if t not in df_negative.ticker.unique().tolist()]

mc_all_stocks = mc_all_stocks[tickers]
bv_all_stocks = bv_all_stocks[tickers]
yearly_ret_all_stocks = yearly_ret_all_stocks[tickers]
monthly_ret_all_stocks = monthly_ret_all_stocks[tickers]

data_ret_1 = data_ret
data_ret_1 = data_ret_1[(data_ret_1.index >= '2012-01-01') & (data_ret_1.index <= '2018-12-31')]

i, m = 0, 0
event_starting_date = pd.Timestamp(year=2012, month=1, day=1)
monthly_rets = []

while event_starting_date < pd.Timestamp(year=2018, month=1,day=1):
    
    event_ending_date = relativedelta(months=12) + event_starting_date
    portfolio_starting_date = event_ending_date
    portfolio_ending_date = portfolio_starting_date + relativedelta(months=1)
    
    ''' divide the stocks into 125 portfolios, based on market capitalization, price to market ratio, and past year return '''

    if m % 12 == 0:
        q1, cutoff1 = pd.qcut(mc_all_stocks.iloc[i], 5, labels=False, retbins=True)
        q1 = pd.DataFrame(q1)
        q1.columns = ['group']
        cutoff1_1 = cutoff1.tolist()

        ''' first level '''
        q1_1 = q1.groupby('group').groups
        q1_1_values = list(q1_1.values())

        q1_2_values = []
        cutoff1_2 = []

        for t in q1_1_values:
            t = t.tolist()
            bv_stocks = bv_all_stocks[t]
            ''' second level '''
            q1_2, cutoff_2 = pd.qcut(bv_stocks.iloc[i], 5, labels=False, retbins=True)
            cutoff1_2.append(cutoff_2.tolist())
            q1_2 = pd.DataFrame(q1_2)
            q1_2.columns = ['group']
            q1_2 = q1_2.groupby('group').groups
            q1_2_values.append(list(q1_2.values()))
    
        q1_2_values = sum(q1_2_values, [])

        cutoff1_3 = []
        q1_3_values = []

        for t in q1_2_values:
            t = t.tolist()
            yr_stocks = yearly_ret_all_stocks[t]
            ''' third level '''
            q1_3, cutoff_3 = pd.qcut(yr_stocks.iloc[i], 5, labels=False, retbins=True)
            cutoff1_3.append(cutoff_3.tolist())
            q1_3 = pd.DataFrame(q1_3)
            q1_3.columns = ['group']
            q1_3 = q1_3.groupby('group').groups
            q1_3_values.append(list(q1_3.values()))
    
        q1_3_values = sum(q1_3_values, [])
        q1_3_values = [q1_3_values[i].tolist() for i in range(125)]


     '''compute the equal-weighted monthly return for each of the 125 portfolios '''
        
        monthly_returns_all_groups = []
    
        for t in q1_3_values:
            df = monthly_ret_all_stocks[t]
            df = df[(df.index < portfolio_starting_date+pd.offsets.DateOffset(years=1)) 
                    & (df.index > portfolio_starting_date)]
            ret = np.mean(df, axis=1).values.tolist()
            #print(len(ret))
            monthly_returns_all_groups.append(ret)
        
        i += 1
    
    ''' find the stocks with negative ESG news within the previous 12 months '''

    df = df_negative[(df_negative.date < event_ending_date)&(df_negative.date >= event_starting_date)]
    tickers = df.ticker.unique().tolist()
    new_tickers = list(set(tickers) & set(mc_esg_stocks.columns) & 
                   set(bv_esg_stocks.columns) & set(yearly_ret_esg_stocks.columns) & set(data_ret_1.columns))


    ''' assign them to each of the 125 portolios based on the precomputed cutoff '''

    mc_esg = pd.DataFrame(mc_esg_stocks.iloc[0][new_tickers])
    mc_esg.columns = ['mc']
    bv_esg = pd.DataFrame(bv_esg_stocks.iloc[0][new_tickers])
    bv_esg.columns = ['bv']
    yr_esg = pd.DataFrame(yearly_ret_esg_stocks.iloc[0][new_tickers])
    yr_esg.columns = ['yr']
    c_esg = mc_esg.join(bv_esg).join(yr_esg)

    c_esg['mc_group'] = pd.cut(c_esg['mc'], bins=cutoff1_1, labels=False, include_lowest=True)

    for i, c in enumerate(cutoff1_2):
        c_esg.loc[c_esg.mc_group == i,'bv_group'] = \
        pd.cut(c_esg[c_esg.mc_group == i]['bv'], bins=c, labels=False, include_lowest=True)
    
    for i in range(5):
        for j in range(5):
            c_esg.loc[(c_esg.mc_group == i) & (c_esg.bv_group == j), 'yr_group'] = \
            pd.cut(c_esg[(c_esg.mc_group == i) &(c_esg.bv_group == j)]['yr'], bins = cutoff1_3[i*5+j], 
                   labels=False, include_lowest=True)

    c_esg['group'] = c_esg['mc_group'] * 25 + c_esg['bv_group'] * 5 + c_esg['yr_group']

    ''' compute the adjusted return for ESG-related stocks '''
    ''' compute the equal-weighted monthly return '''
    rets = data_ret_1[new_tickers][portfolio_starting_date:portfolio_ending_date]
    for t in rets.columns:
        group = c_esg.loc[c_esg.index == t, 'group'].values[0]
        if np.isnan(group):
            rets.drop(columns=t, inplace=True)
        else:
            group_monthly_ret = monthly_returns_all_groups[int(group)]
            rets[t] -= group_monthly_ret[(m % 12)]

    monthly_ret = np.mean(rets, axis=1).values[0]
    monthly_rets.append(monthly_ret)
    
    event_starting_date += relativedelta(months=1)
    m += 1

factors['R'] = monthly_rets
factors['const'] = 1
factors.columns = ['Mkt-RF', 'SMB', 'HML', 'RF', 'MKT', 'Mom', 'R','const']
mod = sm.OLS(factors['R'], factors[['SMB', 'HML', 'MKT','Mom', 'const']])
res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
res.summary()