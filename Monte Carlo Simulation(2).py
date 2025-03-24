import numpy as np
import scipy.stats as stats

def monte_carlo_american_option(S0, K, T, r, sigma, M=100, N=10000, option_type='call'):
    """
    Monte Carlo Least Squares (LSM) pricing of an American option.
    
    Parameters:
    S0: Initial stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility of the underlying asset
    M: Number of time steps
    N: Number of simulations
    option_type: 'call' or 'put'
    
    Returns:
    Estimated option price
    """
    dt = T / M  # Time step
    discount = np.exp(-r * dt)
    
    # Simulating stock price paths
    Z = np.random.randn(N, M)
    S = np.zeros((N, M+1))
    S[:, 0] = S0
    
    for t in range(1, M+1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    # Payoff at maturity
    if option_type == 'call':
        payoff = np.maximum(S[:, -1] - K, 0)
    else:
        payoff = np.maximum(K - S[:, -1], 0)
    
    option_values = payoff.copy()
    
    # Backward induction using regression
    for t in range(M-1, 0, -1):
        in_the_money = S[:, t] > K if option_type == 'call' else S[:, t] < K
        X = S[in_the_money, t]
        Y = option_values[in_the_money] * discount
        
        if len(X) > 0:  # Only proceed if there's something to regress
            # Regression (using Laguerre polynomials as basis functions)
            A = np.column_stack([np.ones_like(X), X, X**2])
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation_values = A @ coeffs
            
            # Immediate exercise values
            immediate_exercise = np.maximum(K - X, 0) if option_type == 'put' else np.maximum(X - K, 0)
            
            # Determine whether to exercise early
            exercise = immediate_exercise > continuation_values
            option_values[in_the_money] = np.where(exercise, immediate_exercise, option_values[in_the_money] * discount)
    
    return np.mean(option_values) * np.exp(-r * dt)

# Example usage
S0 = 100  # Initial stock price
K = 100   # Strike price
T = 1     # Time to maturity 
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility 

price = monte_carlo_american_option(S0, K, T, r, sigma, option_type='put')
print(f"Estimated American Put Option Price: {price:.4f}")
price = monte_carlo_american_option(S0, K, T, r, sigma, option_type='call')
print(f"Estimated American Call Option Price: {price:.4f}")

