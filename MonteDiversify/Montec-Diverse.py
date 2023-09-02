import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import seaborn as sns

# Read in SPX Members:
spx_names = pd.read_csv('spx_names.csv')

# Extract ticker names to a list:
stocks = spx_names['Symbols'].str.split().str[0].tolist()

# Gather price data for ticker names:
start_date = '2019-01-01'
end_date = '2022-02-01'
data_source = 'stooq'
closing_prices = web.DataReader(stocks, data_source, start_date, end_date).Close

# Setting Up Monte Carlo Simulation:
num_stocks_range = list(range(2, 21))  # Number of stocks in each iteration
num_simulations = 10000                # Number of simulations

# Lists to store average returns, volatilities, and Sharpe ratios:
average_returns = []
volatilities = []
sharpe_ratios = []

# Loop for simulations with different numbers of stocks:
for num_stocks in num_stocks_range:
    
    # Lists to store returns, volatilities, and Sharpe ratios for each simulation:
    simulation_returns = []
    simulation_volatilities = []
    simulation_sharpe_ratios = []
    
    # Run simulations for the current number of stocks:
    for _ in range(num_simulations):
        
        # Randomly select num_stocks stock locations:
        random_stock_indices = np.random.choice(len(stocks), num_stocks, replace=False)
        selected_prices = closing_prices.iloc[:, random_stock_indices]
        
        # Calculate random weights for the selected stocks:
        weights = np.random.rand(num_stocks)
        weights /= weights.sum()  # Normalize to sum up to 1
        
        # Perform calculations based on selected stocks and weights:
        expected_portfolio_return = np.sum(weights * selected_prices.mean())
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(selected_prices.cov(), weights)))
        sharpe_ratio = expected_portfolio_return / portfolio_volatility
        
        # Store simulation results:
        simulation_returns.append(expected_portfolio_return)
        simulation_volatilities.append(portfolio_volatility)
        simulation_sharpe_ratios.append(sharpe_ratio)
    
    # Store average results for the current number of stocks:
    average_returns.append(np.mean(simulation_returns))
    volatilities.append(np.mean(simulation_volatilities))
    sharpe_ratios.append(np.mean(simulation_sharpe_ratios))

for _ in range(num_simulations):
        random_stock_indices = np.random.choice(len(stocks), num_stocks, replace=False)
        selected_prices = closing_prices.iloc[:, random_stock_indices]
        weights = np.random.rand(num_stocks)
        weights /= weights.sum()
        
        # Calculate portfolio returns:
        annual_returns = selected_prices.resample('Y').last().pct_change().mean()
        portfolio_return = np.dot(annual_returns, weights)
        simulation_returns.append(portfolio_return)
        
        # Calculate portfolio volatility:
        cov_matrix = selected_prices.pct_change().cov()
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        annual_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)
        simulation_volatilities.append(annual_volatility)
        
        # Calculate Sharpe ratio:
        risk_free_rate = 0.01
        sharpe_ratio = (portfolio_return - risk_free_rate) / annual_volatility
        simulation_sharpe_ratios.append(sharpe_ratio)
    
    # Store average simulation results:
average_returns.append(np.mean(simulation_returns))
volatilities.append(np.mean(simulation_volatilities))
sharpe_ratios.append(np.mean(simulation_sharpe_ratios))

# Plotting the results using Seaborn:
sns.set_theme()

# Returns vs. Number of stocks:
plt.figure(figsize=(8, 5))
sns.lineplot(x=num_stocks_range, y=average_returns)
plt.title('Average Returns vs. Number of Stocks', fontsize=18)
plt.xlabel('Number of Stocks', fontsize=15)
plt.ylabel('Average Return of Portfolio', fontsize=15)

# Volatilities vs. Number of stocks:
plt.figure(figsize=(8, 5))
sns.lineplot(x=num_stocks_range, y=volatilities)
plt.title('Average Volatilities vs. Number of Stocks', fontsize=18)
plt.xlabel('Number of Stocks', fontsize=15)
plt.ylabel('Average Volatility of Portfolio', fontsize=15)

# Sharpe Ratios vs. Number of stocks:
plt.figure(figsize=(8, 5))
sns.lineplot(x=num_stocks_range, y=sharpe_ratios)
plt.title('Average Sharpe Ratios vs. Number of Stocks', fontsize=18)
plt.xlabel('Number of Stocks', fontsize=15)
plt.ylabel('Average Sharpe Ratio of Portfolio', fontsize=15)

plt.show()