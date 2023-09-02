import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the stocks to be used:
stocks = ['AMZN', 'IBM', 'TSLA', 'AAL', 'UAL', 'DAL',
          'MRK', 'MRNA', 'PFE', 'BAC', 'ECL', 'WFC', 'CVX',
          'BK', 'CDW', 'CTSH', 'MCO', 'NOC', 'AAPL', 'ABT', 'AEE',
          'FANG', 'ALK', 'ADI', 'AIZ', 'AKAM', 'AMD', 'AOS',
          'BA', 'AVY', 'CBRE', 'DHI', 'DVA']

# Gather closing prices from Yahoo for more than 3 years:
start_date = '2019-01-01'
end_date = '2022-03-01'
data_source = 'yahoo'
closing_prices = web.DataReader(stocks, data_source, start_date, end_date)['Close']

# Calculate annual returns and covariance matrix:
annual_returns = closing_prices.resample('Y').last().pct_change().mean()
cov_matrix = closing_prices.pct_change().cov()

# Number of simulations and portfolios, and number of weights:
num_simulations = 2500
num_portfolios = 5000
num_weights = len(stocks)

# Create an empty DataFrame to store generated portfolios:
portfolio_results = pd.DataFrame()

# Loop to perform simulations:
for simulation in range(num_simulations):
    # Lists to store portfolio information for each simulation:
    returns, volatilities, weights, sharpe_ratios = [], [], [], []
    
    # Loop to create random portfolios:
    for portfolio in range(num_portfolios):
        # Generate random weights for the portfolio:
        random_weights = np.random.random(num_weights)
        random_weights /= np.sum(random_weights)
        
        # Calculate portfolio metrics:
        portfolio_return = np.dot(random_weights, annual_returns)
        portfolio_volatility = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))
        portfolio_sharpe = portfolio_return / portfolio_volatility
        
        # Store portfolio metrics in respective lists:
        returns.append(portfolio_return)
        volatilities.append(portfolio_volatility)
        weights.append(random_weights)
        sharpe_ratios.append(portfolio_sharpe)
    
    # Store simulation results in the DataFrame:
    portfolio_results[f'Simulation_{simulation+1}'] = [returns, volatilities, weights, sharpe_ratios]

# Show the number of weights that need to be generated:
print(f"Number of weights to be generated: {num_weights}")

# Visualize the distribution of portfolio returns:
plt.figure(figsize=(10, 6))
sns.histplot(returns, bins=50, kde=True)
plt.title('Distribution of Portfolio Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.show()
