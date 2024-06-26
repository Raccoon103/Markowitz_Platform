import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

assets = ['SPY', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

start = '2019-01-01'
end = '2024-04-01'
data = pd.DataFrame()

# Fetch the data for each stock and concatenate it to the `data` DataFrame
for asset in assets:
    raw = yf.download(asset, start=start, end=end)
    raw['Symbol'] = asset
    data = pd.concat([data, raw], axis=0)

# Initialize df and df_returns
df = portfolio_data = data.pivot_table(index='Date', columns='Symbol', values='Adj Close')
df_returns = df.pct_change().fillna(0)



# (1). Implement an equalweighting strategy as dataframe "eqw". Do not include SPY. 
class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]
        
        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)
        
        # You may answer here.
        
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, 'portfolio_weights'):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns['Portfolio'] = self.portfolio_returns[assets].mul(self.portfolio_weights[assets]).sum(axis=1)

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, 'portfolio_returns'):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns

eqw = EqualWeightPortfolio('SPY').get_results()


# (2). Implement a risk parity strategy as dataframe "rp". Do not include SPY.
class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]
        
        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)
        
        # You may answer here.
        
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, 'portfolio_weights'):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns['Portfolio'] = self.portfolio_returns[assets].mul(self.portfolio_weights[assets]).sum(axis=1)

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, 'portfolio_returns'):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns

rp = RiskParityPortfolio('SPY').get_results()


# (3). Implement a Markowitz strategy as dataframe "mv". Do not include SPY.
def mv_opt(R_n, gamma):
    Sigma = R_n.cov().values
    mu = R_n.mean().values
    n = len(R_n.columns)
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('DualReductions', 0)
        env.start()
        with gp.Model(env=env, name = "portfolio") as model:
            
            # You may answer here.
            # Reminder: long only
            
            model.optimize()
            
            # Check if the status is INF_OR_UNBD (code 4)
            if model.status == gp.GRB.INF_OR_UNBD:
                print("Model status is INF_OR_UNBD. Reoptimizing with DualReductions set to 0.")
            elif model.status == gp.GRB.INFEASIBLE:
                # Handle infeasible model
                print("Model is infeasible.")
            elif model.status == gp.GRB.INF_OR_UNBD:
                # Handle infeasible or unbounded model
                print("Model is infeasible or unbounded.")
            
            
            if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                # Extract the solution
                solution = []
                for i in range(n):
                    var = model.getVarByName(f'w[{i}]')
                    #print(f"w {i} = {var.X}")
                    solution.append(var.X)
                
    return solution

class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]
        
        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)
        
        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets].iloc[i-self.lookback:i]
            self.portfolio_weights.loc[df.index[i], assets] = mv_opt(R_n, self.gamma)
        
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, 'portfolio_weights'):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns['Portfolio'] = self.portfolio_returns[assets].mul(self.portfolio_weights[assets]).sum(axis=1)

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, 'portfolio_returns'):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns
    
mv_list = [MeanVariancePortfolio('SPY').get_results(),
           MeanVariancePortfolio('SPY', gamma=100).get_results(),
           MeanVariancePortfolio('SPY', lookback=100).get_results(),
           MeanVariancePortfolio('SPY', lookback=100, gamma=100).get_results(),
           ]


def plot_performance(strategy_list=None):
    # Plot cumulative returns
    fig, ax = plt.subplots()
    
    (1+df_returns['SPY']).cumprod().plot(ax=ax, label='SPY')
    (1+eqw[1]['Portfolio']).cumprod().plot(ax=ax, label='equal_weight')
    (1+rp[1]['Portfolio']).cumprod().plot(ax=ax, label='risk_parity')
    
    if strategy_list!=None:
        for i, strategy in enumerate(strategy_list):
            (1+strategy[1]['Portfolio']).cumprod().plot(ax=ax, label=f'strategy {i+1}')

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

def report_metrics():
    df_bl = pd.DataFrame()
    df_bl['EQW'] = pd.to_numeric(eqw[1]['Portfolio'], errors='coerce')
    df_bl['RP'] = pd.to_numeric(rp[1]['Portfolio'], errors='coerce')
    df_bl['SPY'] = df_returns['SPY']
    for i, strategy in enumerate(mv_list):
        df_bl[f'MV {i+1}'] = pd.to_numeric(strategy[1]['Portfolio'], errors='coerce')
        # You can add your strategy here.

    qs.reports.metrics(df_bl, mode="full", display=True)

# You can use the following to test:
# (1+df_returns).cumprod().plot()
# plot_performance(mv_list)
# plot_allocation(eqw[0])
# plot_allocation(rp[0])
# plot_allocation(mv_list[0][0])
# plot_allocation(mv_list[1][0])
# report_metrics()
# ...


ans_list = [eqw[0], rp[0]]
for mv_i in mv_list:
    ans_list.append(mv_i[0])
        
print(ans_list)
