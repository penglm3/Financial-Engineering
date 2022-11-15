import numpy as np
import pandas as pd
import math

def binomial_pricing(S0,T,sigma,N,r,c,K,c_or_p,Type,n_future_op=0):
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

    # Black-Scholes parameters as input
    """
    S0 = float(input("initial price S0: "))
    T = float(input("time to maturity T (years): "))
    sigma = float(input("annualized volatility sigma: "))
    N = float(input("number of periods N: "))
    r = float(input("continuously compounded interest rate r: "))
    c = float(input("dividend yield c: "))
    K = float(input("strike price K: "))
    c_or_p = float(input("call or put?  1 (call); -1 (put) "))
    Type = input("E (for European) or A (for American)? ")
    """
    print("initial price S0: ", S0)
    print("time to maturity T (years): ",T)
    print("annualized volatility sigma: ",sigma)
    print("number of periods N: ", N)
    print("continuously compounded interest rate r: ", r)
    print("dividend yield c: ",c)
    print("strike price K: ",K)
    print("call or put?  1 (call); -1 (put) ", c_or_p)
    print("E (for European) or A (for American)? ",Type,"\n================================================")
    if n_future_op==0:
        n_future_op = N # when futures and option share the same expiration date
    print("options on futures expiration n_future_op: ", N)

    # binomial model parameters
    u = math.exp(sigma*(T/N)**0.5)
    d = 1/u
    q = (math.exp((r-c)*(T/N))-d)/(u-d)  # risk-neutral probablity
    print("u: ",u)
    print("d: ",d)
    print("risk neutral probability q: ",q,"\n================================================")

    # binomial tree pricing
    len = N+1
    stock = np.zeros([len,len])
    futures = np.zeros([len,len])
    e_option_stock = np.zeros([len,len])
    a_option_stock = np.zeros([len,len])
    e_option_futures = np.zeros([len,len])
    a_option_futures = np.zeros([len,len])

    # stock price
    for i in range(len): # 0 to N(=len-1)
        for j in range(i,len):
            stock[i,j] = S0*d**(i)*u**(j-i)

    # futures price
    futures[:,N] = stock[:,N]
    for j in range(N-1,-1,-1): # (n_future_op-1) to 0
        for i in range(j+1):
            futures[i,j] = q*futures[i,j+1]+(1-q)*futures[i+1,j+1]

    # option (on stock) price
    ## European
    e_option_stock[:,N] = c_or_p*(stock[:,N]-K)
    e_option_stock[:,N][e_option_stock[:,N]<0] = 0

    for j in range(N-1,-1,-1):
        for i in range(j+1):
            e_option_stock[i,j] = (q*e_option_stock[i,j+1]+(1-q)*e_option_stock[i+1,j+1])/math.exp(r*T/N)

    ## American
    a_option_stock[:,N] = c_or_p*(stock[:,N]-K)
    a_option_stock[:,N][a_option_stock[:,N]<0] = 0

    for j in range(N-1,-1,-1):
        for i in range(j+1): # 0 to j
            exercise = max(c_or_p*(stock[i,j]-K),0)
            not_exercise = (q * a_option_stock[i, j + 1] + (1 - q) * a_option_stock[i + 1, j + 1]) / math.exp(r * T / N)
            if exercise>not_exercise:
                print("early exercise at T = {} ".format(j))
            a_option_stock[i,j] = max(exercise,not_exercise)

    # option (on futures) price
    ## European
    e_option_futures[:,n_future_op] = c_or_p*(futures[:,n_future_op]-K)
    e_option_futures[:,n_future_op][e_option_futures[:,n_future_op]<0] = 0
    for j in range(n_future_op-1,-1,-1):
        for i in range(j+1):
            e_option_futures[i,j] = (q*e_option_futures[i,j+1]+(1-q)*e_option_futures[i+1,j+1])/math.exp(r*T/N)

    ## American
    a_option_futures[:,n_future_op] = c_or_p*(futures[:,n_future_op]-K)
    a_option_futures[:,n_future_op][a_option_futures[:,n_future_op]<0] = 0
    for j in range(n_future_op-1,-1,-1):
        for i in range(j+1):
            exercise = max(c_or_p*(futures[i,j]-K),0)
            not_exercise = (q * a_option_futures[i, j + 1] + (1 - q) * a_option_futures[i + 1, j + 1]) / math.exp(r * T / N)
            if exercise>not_exercise:
                print("early exercise at T = {} ".format(j))
            a_option_futures[i,j] = max(exercise,not_exercise)

    print("price movement of stock:\n", stock,"\n================================================")
    print("price movement of futures:\n", futures,"\n================================================")

    if Type == "E":
        print("Euro. option on stock (e_option_stock):\n", e_option_stock,"\n================================================")
        print("Euro. option on futures (e_option_futures):\n", e_option_futures,"\n================================================")
    else:
        print("Amr.option on stock(a_option_stock):\n", a_option_stock,"\n================================================")
        print("Amr.option on futures(a_option_futures):\n", a_option_futures,"\n================================================")

    return stock,futures,e_option_stock,e_option_futures,a_option_stock,a_option_futures

if __name__ == '__main__':
    # binomial_pricing(S0,T,sigma,N,r,c,K,c_or_p,Type,n_future_op=0)
    stock, futures, _, e_option_futures, _, a_option_futures =  binomial_pricing(100,0.5,0.2,10,0.02,0.01,100,1,"E")

    stock, _, e_option_stock, _, _, _ = binomial_pricing(100,0.25,0.23438,10,0.11941,0,100,-1,"E")                                                                                "E")

    # Q1
    stock,_,_,_,a_option_stock1,_ = binomial_pricing(100, 0.25, 0.3, 15, 0.02, 0.01, 110, 1, "A") # 2.60
    a_option_stock1_0 = a_option_stock1[0,0]

    # Q2 & Q3
    stock,_,_,_,a_option_stock2,_ = binomial_pricing(100, 0.25, 0.3, 15, 0.02, 0.01, 110, -1, "A")  # 12.36
    a_option_stock2_0 = a_option_stock2[0, 0]

    # Q4-check put-call parity
    a_option_stock2_0+100*math.exp(-0.01*0.25) == a_option_stock1_0+110*math.exp(-0.02*0.25)

    # Q5
    _, futures, _, _, _, a_option_futures = binomial_pricing(100, 0.25, 0.3, 15, 0.02, 0.01, 110, 1, "A",10)