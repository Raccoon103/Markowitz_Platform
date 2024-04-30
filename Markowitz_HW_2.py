import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

assets = ['SPY', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
data = pd.DataFrame()

# Fetch the data for each stock and concatenate it to the `data` DataFrame
for asset in assets:
    raw = yf.download(asset, start='2012-01-01', end='2024-04-01')
    raw['Symbol'] = asset
    data = pd.concat([data, raw], axis=0)

# Initialize df and df_returns
Bdf = portfolio_data = data.pivot_table(index='Date', columns='Symbol', values='Adj Close')
df = Bdf.loc['2019-01-01':'2024-04-01']



# (4). Create your own strategy here. 
# You can add parameter but please remain "price" and "exclude" unchanged.

class MyPortfolio:
    def __init__(self, price, exclude, ):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]
        
        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=self.price.index, columns=self.price.columns)
        
        # You can add your criteria here.
        
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, 'portfolio_weights'):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns['Portfolio'] = self.portfolio_returns[assets].mul(self.portfolio_weights[assets]).sum(axis=1)

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, 'portfolio_returns'):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


mp = MyPortfolio(df, 'SPY', ).get_results()
Bmp = MyPortfolio(Bdf, 'SPY', ).get_results()



def plot_performance(price, strategy):
    # Plot cumulative returns
    fig, ax = plt.subplots()
    returns = price.pct_change().fillna(0)
    (1+returns['SPY']).cumprod().plot(ax=ax, label='SPY')
    (1+strategy[1]['Portfolio']).cumprod().plot(ax=ax, label=f'MyPortfolio')

    ax.set_title('Cumulative Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    plt.show()
    return None

def plot_allocation(df_weights):
    df_weights = df_weights.fillna(0).ffill()
    
    # long only
    df_weights[df_weights < 0] = 0

    # Plotting
    fig, ax = plt.subplots()
    df_weights.plot.area(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Allocation')
    ax.set_title('Asset Allocation Over Time')
    plt.show()
    return None

def report_metrics(price, strategy, show=False):
    df_bl = pd.DataFrame()
    returns = price.pct_change().fillna(0)
    df_bl['SPY'] = returns['SPY']
    df_bl[f'MP'] = pd.to_numeric(strategy[1]['Portfolio'], errors='coerce')

    qs.reports.metrics(df_bl, mode="full", display=show)
    
    sharpe_ratio = qs.stats.sharpe(df_bl)
    
    return sharpe_ratio

# You can use the following to test:
# (1+df.pct_change().fillna(0)).cumprod().plot()
# plot_performance(df, mp)
# plot_allocation(mp[0])

# (1+Bdf.pct_change().fillna(0)).cumprod().plot()
# plot_performance(Bdf, Bmp)
# plot_allocation(Bmp[0])

# report_metrics(df, mp, show=True)
# report_metrics(Bdf, Bmp, show=True)


print(report_metrics(df, mp)[1] > 1)
print(report_metrics(Bdf, Bmp)[1] > report_metrics(Bdf, Bmp)[0])
