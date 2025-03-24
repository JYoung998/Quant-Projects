import numpy as np

def american_binomial_option_pricer(S, K, T, r, sigma, n, option_type="call"):
    """
    Price an American option using the binomial tree model (CRR model)
    
    Parameters:
    S: float  -> Initial stock price
    K: float  -> Strike price
    T: float  -> Time to maturity (in years)
    r: float  -> Risk-free interest rate (annualized)
    sigma: float -> Volatility of the underlying asset
    n: int -> Number of time steps
    option_type: str -> "call" or "put"
    
    Returns:
    float: The price of the American option
    """
    dt = T / n  # Time step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
    
    # Stock price tree
    stock_tree = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Option value tree
    option_tree = np.zeros((n + 1, n + 1))
    if option_type == "call":
        option_tree[:, n] = np.maximum(stock_tree[:, n] - K, 0)
    elif option_type == "put":
        option_tree[:, n] = np.maximum(K - stock_tree[:, n], 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Backward induction to calculate option price
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            intrinsic_value = max(stock_tree[j, i] - K, 0) if option_type == "call" else max(K - stock_tree[j, i], 0)
            option_tree[j, i] = max(continuation_value, intrinsic_value)
    
    return option_tree[0, 0]

# Example usage:
S = 100  # Initial stock price
K = 100  # Strike price
T = 1  # Time to maturity 
r = 0.07  # Risk-free rate 
sigma = 0.24  # Volatility 
n = 100  # Number of steps

price_call = american_binomial_option_pricer(S, K, T, r, sigma, n, option_type="call")
price_put = american_binomial_option_pricer(S, K, T, r, sigma, n, option_type="put")

print(f"American Call Option Price: {price_call:.4f}")
print(f"American Put Option Price: {price_put:.4f}")


