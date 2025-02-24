# ##############################################################################
#      Author: Edward E. Daisey
#       Class: Matrix Theory
#   Professor: Dr. Cutrone
#        Date: 23 February 2025
#       Title: Monte Carlo Simulation for a Value Portfolio vs. The S&P 500.
# Description: This code presents two portfolios over a ten year period of time.
#              This corresponds to approximately 2520 trading days. The first
#              portfolio -- the value portfolio -- hold ten value stocks
#              outside the S&P 500.  The second portfolio -- the S&P 500 portfolio --
#              is comprised of VOO.  Both portfolios have an inital allocation
#              of $10,000.  This code uses a log-normal price process driven by
#              a PCG64-based pseudo-random number generator for daily returns.
#              Floor rounding is used to ensure that only whole shares are purchased.
# ##############################################################################


# ################################ Packages ###################################
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.random import PCG64, Generator
# #############################################################################


# ############################## Constants ####################################
TOTAL_DAYS = 2520       # Represents approximately 10 years of trading.
NUM_SIMULATIONS = 100   # Total number of Monte Carlo simulations.
SEED_VALUE = 10000      # Initial Seed.

# ############### Value Portfolio ###############
# Below is the list of stocks in the value portfolio. Each stock is allocated
# $1,000 in order to purchase whole shares in the company.
value_tickers = [
    "CNK",  # Cinemark Holdings
    "HAIN", # Hain Celestial Group
    "LAMR", # Lamar Advertising Company
    "MAC",  # Macerich Company
    "MG",   # Mistras Group
    "PEBO", # Peoples Bancorp
    "PLPC", # Preformed Line Products
    "SIG",  # Signet Jewelers
    "THS",  # TreeHouse Foods
    "VAL"   # Valaris Limited
]

# Current prices (USD) for the aforementioned stocks. Taken on 22 February 2025. 
value_current_prices = np.array([27.45, 4.18, 121.91, 19.82, 9.89, 
                                 31.78, 134.32, 52.75, 30.58, 41.78]) 

# Daily log-return and volatility for the value portfolio.
MU_VALUE = 0.0005725   # Feel Free To Change THis To Match Your Portfilio!
SIGMA_VALUE = 0.0126
# ###############################################

# ############### S&P 500 Portfolio ###############
# S&P 500 simulation details. We use VOO at $550.
sp500_ticker = "VOO"
sp500_current_price = 550.0

# Typical daily log-return and volatility for the S&P 500.
MU_SP500 = 0.0004141
SIGMA_SP500 = 0.00968
#################################################

# Rationale for mu (daily log-return) and sigma (daily volatility) for both portfolios:
# ------------------------------------------------------------------------------------
# 1) Value portfolio (mu=0.0005725, sigma=0.0126):
#    - Annual return ≈ 15.52% → mu_daily ≈ ln(1.1552) / 252 ≈ 0.0005725.
#    - Annual vol ≈ 20% → sigma_daily ≈ 0.20 / √252 ≈ 0.0126.
#
# 2) S&P 500 (mu=0.0004141, sigma=0.00968):
#    - Annual return ≈ 11% → mu_daily ≈ ln(1.11) / 252 ≈ 0.0004141.
#    - Annual vol closer to ~15.38% → sigma_daily ≈ 0.1538 / √252 ≈ 0.00968.
# #############################################################################


# ############################## Function 1 ###################################
# Function Name: SimulateStockPath
# Function Purpose: Simulate a single stock path over TOTAL_DAYS using daily 
#                   log-returns.
# Function Input:
#    initial_price (float) - The starting stock price.
#         total_days (int) - The number of trading days to simulate.
#               mu (float) - The average daily log-return.
#            sigma (float) - The standard deviation of the daily log-return.
#          rng (Generator) - PCG64-based random number generator (rng) instance.
# Function Output:
#    A numpy array of length total_days + 1 representing daily stock prices.
def SimulateStockPath( initial_price, total_days, mu, sigma, rng ):
    daily_returns = rng.normal( loc = mu, scale = sigma, size = total_days)
    log_cumsum    = np.cumsum( daily_returns )

    path          = np.zeros( total_days + 1 )
    path[0]       = initial_price
    path[1:]      = initial_price * np.exp( log_cumsum )
    return path
# #############################################################################


# ############################## Function 2 ###################################
# Function Name: SimulateValuePortfolio
# Function Purpose: Model a 10-stock value portfolio, each allocated $1,000,  
#                   using floor rounding to purchase whole shares.
# Function Input:
#    prices (numpy array) - The current prices for the 10 value stocks.
#        total_days (int) - The number of trading days to simulate.
#              mu (float) - The average daily log-return for these stocks.
#           sigma (float) - The daily volatility for these stocks.
#         rng (Generator) - PCG64-based random number generator instance.
# Function Output:
#    A tuple: (portfolio_values, stock_paths)
#    - portfolio_values (1D array): length total_days+1, daily portfolio value.
#    -      stock_paths (2D array): shape (num_assets, total_days+1) with each 
#                                   asset's path.
def SimulateValuePortfolio( prices, total_days, mu, sigma, rng ):
    # Determine Number of Assets (n = 10) & Shares:
    num_assets = len( prices )
    shares     = np.floor( 1000.0 / prices )

    # Initialize Stock Price Paths & Generate Daily Random Log-Returns For Each Stock:
    stock_paths       = np.zeros( ( num_assets, total_days + 1 ) )
    daily_returns_all = rng.normal(  loc = mu, 
                                   scale = sigma,
                                    size = ( num_assets, total_days ) )
    
    # Compute Cumulativ Log-Returns:
    log_cumsum_all    = np.cumsum( daily_returns_all, axis = 1 )

    # Simulate Stock Price Paths:
    for i in range( num_assets ):
        stock_paths[ i, 0 ]  = prices[ i ]
        stock_paths[ i, 1: ] = prices[ i ] * np.exp( log_cumsum_all[ i ] )

    # Compute Total Portfolio Value at Each Time Step:
    portfolio_values = np.sum( shares[ :, None ] * stock_paths, axis = 0 )
    return portfolio_values, stock_paths
# #############################################################################


# ############################## Function 3 ###################################
# Function Name: SimulateSP500Portfolio
# Function Purpose: Model a S&P 500 (i.e., VOO) portfolio, allocated $10,000,  
#                   using floor rounding to purchase whole shares.
# Function Input:
#    current_price (float) - The initial price of VOO (i.e., $550).
#         total_days (int) - The number of trading days to simulate.
#         mu_bench (float) - The average daily log-return for VOO.
#      sigma_bench (float) - The daily volatility for VOO.
#          rng (Generator) - PCG64-based random number generator instance.
# Function Output:
#    A tuple: (portfolio_values, sp500_path)
#    - portfolio_values (1D array): length total_days+1, daily portfolio value.
#    -       sp500_path (1D array): length total_days+1, simulated daily 
#                                   VOO prices.
def SimulateSP500Portfolio( current_price, total_days, mu_bench, sigma_bench, rng ):
    # Determine Number of Shares:
    shares_sp500     = np.floor( 10000.0 / current_price )
    
    # Simulate VOO Price Path:
    sp500_path       = SimulateStockPath( current_price, total_days, mu_bench, sigma_bench, rng )
    
    # Compute Portfolio Value Over Time:
    portfolio_values = sp500_path * shares_sp500
    return portfolio_values, sp500_path
# #############################################################################


# ############################## Function 4 ###################################
# Function Name: MonteCarloSim (Particularly Geometric Brownian Motion-based Monte Carlo)
# Function Purpose: Run multiple simulations for the value portfolio vs. S&P 500.
# Function Input:
#                   num_sims (int) - Number of Monte Carlo runs.
#                 total_days (int) - Number of trading days to simulate.
#       value_prices (numpy array) - Prices for the value stocks.
#    mu_value, sigma_value (float) - Log-return & volatility for the value stocks.
#              sp500_price (float) - The initial VOO price.
#    mu_sp500, sigma_sp500 (float) - Log-return & volatility for VOO.
# Function Output:
#    A tuple: (final_value_vals, final_sp500_vals)
#    - final_value_vals (1D array): final day values for the value portfolio across runs.
#    - final_sp500_vals (1D array): final day values for VOO across runs.
def MonteCarloSim(    num_sims, total_days,
                  value_prices,   mu_value,  sigma_value, 
                   sp500_price,   mu_sp500,  sigma_sp500 ):
    
    # Initializing Storage For Final Portfolio Values:
    final_value_vals = np.zeros( num_sims )
    final_sp500_vals = np.zeros( num_sims )

    # Random Number Generator:
    rng = Generator( PCG64( SEED_VALUE ) )

    # Monte Carlo Loop & Simulating Portfolios:
    for i in range( num_sims ):  # Each iteration represent one 10-year stock market scenario.
        val_port, _   = SimulateValuePortfolio( value_prices, total_days, mu_value, 
                                                 sigma_value, rng )
        sp500_port, _ = SimulateSP500Portfolio(  sp500_price, total_days, mu_sp500, 
                                   sigma_bench = sigma_sp500, rng = rng )
        
        # Storing Final Portfolio Values:
        final_value_vals[ i ] =   val_port[ -1 ]
        final_sp500_vals[ i ] = sp500_port[ -1 ]

    return final_value_vals, final_sp500_vals
# #############################################################################


# ############################## Function 5 ###################################
# Function Name: Main
# Function Purpose: Coordinate the simulation, display results, and plotting of data.
def Main():
    # Run Monte Carlo Sim:
    final_val, final_sp5 = MonteCarloSim(
             NUM_SIMULATIONS, TOTAL_DAYS,
        value_current_prices, MU_VALUE, SIGMA_VALUE,
         sp500_current_price, MU_SP500, SIGMA_SP500
    )

    # Compute & Output Key Statistics
    mean_val         = np.mean( final_val )
    std_val          = np.std(  final_val )
    mean_sp5         = np.mean( final_sp5 )
    std_sp5          = np.std(  final_sp5 )
    median_sp5_final = np.median(final_sp5)
    beat_count       = np.sum( final_val > median_sp5_final )  
    beat_pct         = ( beat_count / NUM_SIMULATIONS ) * 100
    print( f"\n====================== Monte Carlo Results ({NUM_SIMULATIONS} Runs, 10 Years) =====================")
    print( f"{'Metric':<35}{'Value Portfolio':>20}{'S&P 500 (VOO)':>25}" )
    print( "-" * 85)
    print( f"{'Final Value (Mean)':<35}{mean_val:>20.2f}{mean_sp5:>25.2f}" )
    print( f"{'Final Value (Std. Dev.)':<35}{std_val:>20.2f}{std_sp5:>25.2f}" )
    print( f"{'Outperformance Frequency (%)':<35}{beat_pct:>20.2f}\n" )
    # Note: Outperformance Frequency (%) measures how often the Value Portfolio
    #       ends with a higher final value than the *median* of the S&P 500
    #       portfolio.  E.g., if 100 simulations, then a 99% outperformance
    #       frequency means that the Value Portfolio ended higher than the median
    #       S&P 500 final value in 99 out of 100 runs (or 99% of the time).

    # Example Output:
    # ====================== Monte Carlo Results (100 Runs, 10 Years) =====================
    # Metric                                  Value Portfolio            S&P 500 (VOO)
    # -------------------------------------------------------------------------------------
    # Final Value (Mean)                             51131.91                 29491.18
    # Final Value (Std. Dev.)                          11322.09                 15893.19
    # Outperformance Frequency (%)                     100.00

    # Histogram:
    plt.figure( figsize = ( 12, 7 ) )
    plt.title( 'Final Portfolio Value Distribution', fontsize = 14 )
    plt.xlabel( 'Portfolio Value ($)', fontsize = 12 )
    plt.ylabel( 'Frequency', fontsize = 12 )  
    bin_edges = np.linspace( min( final_val.min(), final_sp5.min()), 
                             max( final_val.max(), final_sp5.max()), 
                             31)  # 30 Bins
    plt.hist(final_sp5, bins = bin_edges, alpha = 0.7, label = 'S&P 500 Portfolio', color = 'red' )
    plt.hist(final_val, bins = bin_edges, alpha = 0.7, label = 'Value Portfolio',   color = 'blue' )
    plt.legend( fontsize = 12,   loc = 'upper right', edgecolor = 'black', 
                fancybox = True, framealpha = 1, title_fontsize = 12 )
    plt.grid( True )
    plt.show()


    # Monte Carlo Plot:
    plt.figure( figsize=( 12, 7 ) )
    plt.title( 'Sample 10-Year Monte Carlo Paths: Value vs. S&P 500', fontsize = 14 )
    plt.xlabel( 'Trading Day', fontsize = 12 )
    plt.ylabel( 'Portfolio Value ($)', fontsize = 12 )
  
    # Random Number Generator For Monte Carlo Plot:    
    rng_sample = Generator( PCG64( int( time.time() ) ) )
    
    # Initialize Lists For Storing Portfolio Paths For Plot:
    sample_val_paths = []
    sample_sp5_paths = []
    
    # Simulate & Plot Monte Carlo Paths For Plot:
    for _ in range( NUM_SIMULATIONS ):
        val_port, _ = SimulateValuePortfolio( value_current_prices, TOTAL_DAYS, MU_VALUE, SIGMA_VALUE, rng_sample )
        sp5_port, _ = SimulateSP500Portfolio(  sp500_current_price, TOTAL_DAYS, MU_SP500, SIGMA_SP500, rng_sample )
        sample_val_paths.append( val_port )
        sample_sp5_paths.append( sp5_port )
    for path in sample_val_paths:
        plt.plot( path, color = 'blue', alpha = 0.25,
                 label = 'Value Portfolio' if 'Value Portfolio' not in plt.gca().get_legend_handles_labels()[ 1 ] else "" )
    for path in sample_sp5_paths:
        plt.plot( path, color = 'red', alpha = 0.25,
                 label = 'S&P 500 Portfolio' if 'S&P 500 Portfolio' not in plt.gca().get_legend_handles_labels()[ 1 ] else "" )
    
    # Compute & Plot Median Price Path For Plot:
    median_val_path = np.median( sample_val_paths, axis = 0 )
    median_sp5_path = np.median( sample_sp5_paths, axis = 0 )
    plt.plot( median_val_path, color = 'blue',    linewidth = 3, label = 'Median Value Portfolio' )
    plt.plot( median_sp5_path, color = '#8B0000', linewidth = 3, label = 'Median S&P 500 Portfolio' )
    
    # Finalize Monte Carlo Plot:
    plt.legend()
    plt.grid( True )
    plt.show()
# #############################################################################


# ############################## Main Execution ###############################
if __name__ == "__main__":
    Main()
# #############################################################################
