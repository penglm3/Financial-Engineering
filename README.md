# Financial-Engineering


[binomial_model.py](https://github.com/penglm3/Financial-Engineering/blob/main/binomial_model.py): pricing European & American (call and put) options on stock and futures using binomial trees
```python
binomial_pricing(S0,T,sigma,N,r,c,K,c_or_p,Type,n_future_op=0):
    """
    pricing European & American options on stock and futures by binomial models
    parameters are in the Black-Scholes format
    :param S0: initial price of the underlying asset, float
    :param T: time to maturity (years), float
    :param sigma: annualized volatility, float
    :param N: number of periods in the binomial tree (before futures expires), float
    :param r: continuously compounded interest rate, float
    :param c: dividend yield, float
    :param K: strike price, float
    :param c_or_p: call or put?  1 (call); -1 (put), int
    :param Type: E (for European) or A (for American)?, string
    :param n_future_op: maturity of options on futures (different from N), int, default = N
    :return: matrices of the stock price (stock), futures price (futures),
        Euro. option on stock (e_option_stock), Euro. option on futures (e_option_futures),
        Amr. option on stock (a_option_stock), Amr. option on futures (a_option_futures)
    """
```
