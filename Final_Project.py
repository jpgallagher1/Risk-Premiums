"""

John Gallagher
X433.3-007
Alex Iliev

Final Project

Sections:
1. Proposal
2. Background
3. Code Outline
4. Code

---
1. Class Final Project Proposal:

This project will extend some of my existing knowledge in quantitative finance 
into a new programming language.  

Create a simple trading algorithm, that takes inputs from a stock's price.  
After creating the signal, with no look-ahead bias, track the simulated 
performance of the algorithm. Using summary statisics, summarize, and visualize 
the performance of the algorithm. 

This is an analog of a packag in R (in the public domain) that was developed for
trading financial instruments.  I want to replicate some of the functionality 
using Python. 
---

2. Background: 
The some R packages for quant strategy development are 'quantmod', 'quantstrat',
'QuantTools', 'PerformanceAnalytics' etc. 

Some packages that are for data frames, tables, timeseries: 
'data.table', 'dplyr', 'xts', 'zoo' etc.  

Usual package for data visualization: 'ggplot2' (analogous to matplotlib)

The packages are objects and functions developed to allow for easy calculations
on common financial unstruments.  The calculations are specific to quantitative 
financial techniques within industry and academia.  

---
3. Code Outline
    A. Import financial data
        i.  S&P 500 Financials, Ticker list of companies
        ii. Stock Price: Open, High, Low, Close, Volume (OHLC) 
    B. Calculate additional variables and statistics
        i.   Return [(close_t/close_t-1) -1]
        ii.  Trading Signal
    C. Create holdings (if long: 1, if short -1, if neutral: 0)
        i.   Make sure holdings are lagged (to avoid look ahead bias)
        ii.  Rescale weights so net leverate == 2x (1x long, 1x short.)
    D. Generate performance of strategy
    E. Summary Statistics on performance of strategy
        i.   Cumulative Value Add Plot (CVA)
        ii.  Return and Volatiltiy Annualized
        iii. Rolling Volatitlity
        iv.  Rolling Sharpe Ratio
        v.   Drawdown Plots showing peak to trough for the worst periods. 
        

"""
import numpy as np
from scipy import linalg
import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import pyfolio as pf
import quandl

quandl.ApiConfig.api_key = 'YOURAPIKEY'

# A. Import Financial Data
#   i. S&P 500: GICS == Financials, company ticker list

def spxConstituents(wiki_url):
    """
    Pull SPX constituents and some additional data from wikipedia. Replaces string dates as datetime objects

    input: str (URL) 
    output: pd.Dataframe (wiki table)

    Dataframe column names: 
        (['Security', 'Symbol', 'SEC filings', 'GICS Sector', 'GICS Sub Industry', 'Headquarters Location', 'Date first added', 'CIK', 'Founded'])
    """
    SPX_list, SPX_chng = pd.read_html(wiki_url, header=0)
    SPX_list['Date first added'] = pd.to_datetime(SPX_list['Date first added'])
    del SPX_chng
    return SPX_list

def spxFinancials(myDataFrame):
    """
    Returns GICS == financials from SPX dataframe

    input: pd.Dataframe 
    output: subset of pd.Dataframe
    """
    financialSubset = myDataFrame[myDataFrame['GICS Sector'] == 'Financials']
    return financialSubset

def removeYoungTickers(myDataFrame, startDate):
    """
    Checks data frame for tickers that don't have enough history to be included in the analysis.

    NaT means they are original constituents of the S&P 500 and should be included

    input: pd.DataFrame, datetime.datetime 
    output: pd.Dataframe(subset)
    """
    for eachDate in myDataFrame['Date first added']:
        if pd.isnull(eachDate):
            eachDate = datetime.datetime.min
        else: pass
    return myDataFrame[myDataFrame['Date first added'] < startDate]

def tickers(myDataFrame):
    """
    Return a list of ticker strings for use in DataReader data pull
    """
    return myDataFrame['Symbol'].tolist()


# A. Import Financial Data
#   ii. Historic Data, 10 years of price history
#        Price, Open, High, Low, Close, Volume
def pulldata(ticker_list, source, start, end):
    """
    returns dictionary of data frames for companies, {ticker: OHLCV dataframe}

    input: list(str), str, datetime, datetime
    output: Dictionary {str : pd.Dataframe}
    """    
    d = {}
    for company in ticker_list:
        d[company] = web.DataReader(company, source, start, end)
    return d

# B. Calculate additional variables and statistics
#     i.   Return [(close_t/close_t-1) -1]

def genTickerReturns(ohlcv_dict, ticker_list):
    """
    Creates a dataframe of returns indexed by (rows, columns) = (dates, tickers)

    input: ohlcv_dictionary{'str':pd.Dataframe}, list['str']
    output: pd.DataFrame
    """
    tkr_returns_dict = {}
    tkr_returns = pd.DataFrame()
    for a_ticker in ticker_list:
        tkr_returns_dict[a_ticker] = ohlcv_dict[a_ticker].pct_change()['close']
    tkr_returns = pd.DataFrame.from_dict(tkr_returns_dict)
    tkr_returns.index = pd.to_datetime(tkr_returns.index)
    return tkr_returns

def rollingVol(aDataframe, windowsize = 22):
    return aDataframe.rolling(windowsize).std()


# B. Calculate additional variables and statistics
#     vi.  Trading Signal


def momentumSignal(returnsDataframe, window = 33):
    """
    generates a simple momentum signal that is the cumulative return of the past 33 trading days (1.5 mo)

    input: stock return dataframe
    output: stock signal dataframe
    """
    signaldf = returnsDataframe.rolling(window).apply(lambda x: np.prod(1+x)-1)
    return signaldf

# C. Create holdings (if long: 1, if short -1, if neutral: 0)
#     i.   Make sure holdings are lagged (to avoid look ahead bias)
#     ii.  Rescale weights so net leverate == 2x (1x long, 1x short.)

def centeredDF(aDataframe):
    return aDataframe.subtract(aDataframe.mean(axis = 1), axis = 0)

def setGrossLeverage(holdingsDataframe, grossLeverage = 1):
    """
    takes current holdings and rescales weights 
    """
    grossWts = abs(holdingsDataframe).sum(axis=1)
    return grossLeverage * holdingsDataframe.divide(grossWts, axis=0)


def longShortN(signalDF, n):
    """
    Takes trading signal dataframe, ranks the signal strength, then goes long the N highest, and shorts the N lowest ranks.

    input: Trading Signal Dataframe
    output: n,long (1) and n, short (-1)
    """
    dfWidth = signalDF.shape[1]
    if dfWidth < (n*2): 
        return print("The width of the dataframe is less than 'n'. ")
    rankedDF = signalDF.rank(axis = 1, method = 'first')
    outputDF = rankedDF

    outputDF[outputDF <(n +1)] = -1
    outputDF[outputDF>(dfWidth - n)] = 1
    outputDF[(outputDF<(dfWidth - n + 1)) & (outputDF > n)] = 0

    return outputDF

def genPortReturns(holdingReturnsDF):
    return holdingReturnsDF.sum(axis = 0)

def sharpe(returns, rf:float, days=252):
 volatility = returns.std() * np.sqrt(days) 
 anualizedReturns = (1+returns.mean())**252 -1
 sharpe_ratio = (anualizedReturns - rf) / volatility
 return pd.DataFrame({'Return':anualizedReturns, 'Vol':volatility, 'Sharpe': sharpe_ratio}).T

def information_ratio(returns, benchmark_returns, days=252):
 return_difference = returns - benchmark_returns 
 volatility = return_difference.std() * np.sqrt(days) 
 information_ratio = return_difference.mean() / volatility
 return information_ratio


# ---
# 3. Code Outline
#     A. Import financial data
#         i.  S&P 500 Financials, Ticker list of companies
#         ii. Stock Price: Open, High, Low, Close, Volume (OHLC) 
#     B. Calculate additional variables and statistics
#         i.   Return [(close_t/close_t-1) -1]
#         ii.  Trading Signal
#     C. Create holdings (if long: 1, if short -1, if neutral: 0)
#         i.   Make sure holdings are lagged (to avoid look ahead bias)
#         ii.  Rescale weights so net leverate == 2x (1x long, 1x short.)
#     D. Generate performance of strategy
#     E. Summary Statistics on performance of strategy
#         i.   Cumulative Value Add Plot (CVA)
#         ii.  Return and Volatiltiy Annualized
#         iii. Rolling Volatitlity
#         iv.  Rolling Sharpe Ratio
#         v.   Drawdown Plots showing peak to trough for the worst periods. 

# Now do run the functions to get the output
################
#     A. Import financial data
# ---
SPX_list_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
quandl = 'quandl' # NB: you need your own API Key to pull data from here
iex = 'iex' # NB: Now broken due to migrating their API and sunsetting this version. 
start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2019, 1, 1)
companies = spxConstituents(SPX_list_wiki_url)
fin_companies = removeYoungTickers(spxFinancials(companies), start)
# ---
fin_tickers = tickers(fin_companies)
bench_tickers = ['SPY', 'IYG']
# ---
tickerDF = pulldata(fin_tickers, 'iex', start, end)
benchDF = pulldata(bench_tickers, iex, start,end)
# ---
# B. Calculate Addtional variables
tickerReturns = genTickerReturns(tickerDF, fin_tickers)
benchReturns = genTickerReturns(benchDF, bench_tickers)
# ---
tradingSignal = momentumSignal(tickerReturns, window=126)

# C. Create holdings
holdings = longShortN(tradingSignal, 18)
scaledholdings = setGrossLeverage(holdings, grossLeverage= 2)
scaledholdings = scaledholdings.shift(1)

# D. Generate performance of strategy
portReturns = scaledholdings*tickerReturns
portTotReturn = portReturns.sum(axis=1)
blend = portTotReturn*.5 + benchReturns['SPY']*.5
benchReturns['50/50 SPY Port'] = blend
# ---
allreturns = benchReturns
allreturns['Port'] = portTotReturn
# ---
portcumret = (1+portTotReturn.tail(1205-126)).cumprod()
benchcumret = (1+benchReturns.tail(1205-126)).cumprod()
# ---
allcumReturns = benchcumret
allcumReturns['Port'] = portcumret


#  E. Summary Statistics on performance of strategy
xtickrange = pd.date_range(allcumReturns.index[0], end, freq="Q")
ytickrange = np.arange(0.75, 1.85, 0.1)
ax = allcumReturns.plot(title = "Pure and Blended Strategy vs Benchmarks")
ax.set(xlabel = "Date", ylabel = "Cumulative Return", 
       xticks = xtickrange,
       yticks = ytickrange)
ax.yaxis.grid(True)
legend = ax.legend(loc = "lower right")
plt.savefig('Pure and Blended Strategy vs Benchmarks.png', bbox_inches='tight')
plt.close()

# Drawdown Charts
ax1 = pf.plot_drawdown_underwater(benchReturns['Port'])
ax1.set(title = "Strategy portfolio Drawdowns", xlabel = "Date", ylabel = "Percentage Down from Peak", xticks = xtickrange)
plt.savefig('Strategy Drawdowns.png', bbox_inches='tight')
plt.close()

ax2 = pf.plot_drawdown_underwater(benchReturns['50/50 SPY Port'])
ax2.set(title = "Blended portfolio Drawdowns", xlabel = "Date", ylabel = "Percentage Down from Peak", xticks = xtickrange)
plt.savefig('Blended Drawdowns.png', bbox_inches='tight')
plt.close()

ax3 =pf.plot_drawdown_underwater(benchReturns['SPY'])
ax3.set(title = "Benchmark Drawdowns", xlabel = "Date", ylabel = "Percentage Down from Peak", xticks = xtickrange)
plt.savefig('Benchmark Drawdowns.png', bbox_inches='tight')
plt.close()

# Rolling Charts
ax4 = pf.plot_rolling_returns(benchReturns['50/50 SPY Port'], factor_returns=benchReturns['SPY'])
ax4.set(title = "Blended vs Benchmark Rolling Returns", xlabel = "Date", ylabel = "Return in %", xticks = xtickrange)
plt.savefig('Blended vs Benchmar Rolling Returns.png')
plt.close()

ax5 = pf.plot_rolling_volatility(benchReturns['50/50 SPY Port'], factor_returns=benchReturns['SPY'])
ax5.set(title = "Blended vs Benchmark Rolling Volatiltiy", xlabel = "Date", ylabel = "One standard deviation in %", xticks = xtickrange)
plt.savefig('Blended vs Benchmark Rolling Volatiltiy.png')
plt.close()

ax6 = pf.plot_rolling_sharpe(benchReturns['50/50 SPY Port'], factor_returns=benchReturns['SPY'])
ax6.set(title = "Blended vs Benchmark Rolling Sharpe Ratio", xlabel = "Date", ylabel = "Ratio of Return:Volatility", xticks = xtickrange)
plt.savefig('Blended vs Benchmark Rolling Sharpe.png')
plt.close()

# Summary Table
sharpeTable = sharpe(allreturns, 0)
sharpeTable.to_csv('Strategy and Blend vs Benchmarks.csv')

