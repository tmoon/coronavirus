import numpy as np 
import scipy as sp
import pandas as pd
from scipy import stats, optimize, interpolate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
import time

from e_step import update_data, compute_S, compute_E

np.seterr(all='ignore')
warnings.filterwarnings('ignore')

def train(C, D, N, inits, priors, rand_walk_stds, t_ctrl, tau, n_iter, n_burn_in, m, bounds):
    """
    C = the number of cases by date of symptom onset
    D = the number of cases who are removed (dead or recovered)
    N = total population
    inits is a list of:
        s0 = S(0): number of suspected individuals at t=0
        e0 = E(0): number of exposed individuals at t=0
        i0 = I(0): number of infected individuals at t=0
             (this is called a in paper)
    priors = list of gamma prior parameters for four model parameters.
             The parameters are:
                beta: uncontrolled transmission rate
                q: parameter for time dependent controlled transmission rate
                g: 1/mean incubation period
                gamma: 1/mean infectious period

    rand_walk_stds = proposal dist for MCMC parameter update is a normal distribution
                     with previous sample as mean and standard deviation from
                     rand_walk_stds. There is one std for each of the four model
                     parameters. 

                     TODO: For faster convergence, the stds should be tuned such that 
                     mean acceptance probability is in between 0.23 and 0.5.

    t_end = end of observation
    t_ctrl = day of intervention
    tau = end of epidemics in days > t_end
    
    n_iter = number of iterations
    n_burn_in = burn in period for MCMC
    
    m = total number of infected individuals throughout the course of the disease = sum(B)
    

    returns: the distribution of B and params. They can be used later to calculate R0 and extrapolate

    """
    t_end = len(C)
    assert t_end < tau
    assert t_ctrl < tau
    assert len(C) == len(D)
    assert inits[0] >= m # s0 >= m
    assert n_burn_in < n_iter

    # initialize
    B = np.zeros((t_end,))
    B[0] = m
    assert np.sum(B) == m

    # initialize model parameters
    params = [0.2, 0.2, 0.2, 0.2]
    beta, q, g, gamma = params
    s0, e0, i0 = inits
    epsilon = 1e-16
    
    # initialize I, S, E, P
    I = compute_I(inits[2], t_end, C, D)
    S = compute_S(s0, t_end, B)
    E = compute_E(e0, t_end, B, C)
    P = compute_P(transmission_rate(beta, q, t_ctrl, t_end), I, N)
    
    # rep invariants
    # I, S, E should be non-negative
    assert (I >= 0).all()    
    assert (S >= 0).all()
    assert (E >= 0).all()
    assert (E + I > 0).all()
    # P is a list of binomial parameters
    assert (1 >= P).all() and (P >= 0).all()

    # initialize B and params
    print(f"n_burn_in:{n_burn_in}")
    # to show final statistics about params
    saved_params = []
    saved_R0ts = []

    start_time = time.time()
    t0 = start_time
    t1 = start_time
    for i in range(n_iter):
        # MCMC update for B, S, E
        B, S, E, log_prob_new, log_prob_old = update_data(B, C, D, P, I, S, E, inits, params, N, t_end, t_ctrl, m, epsilon)

        assert np.round(np.sum(B)) == m
        # MCMC update for params and P
        # I is fixed by C and D and doesn't need to be updated
        params, P, R0t, log_prob_new, log_prob_old = update_params(B, C, D, P, I, S, E, inits, 
                                            params, priors, rand_walk_stds, N, t_end, t_ctrl, epsilon, 
                                            bounds)

        if i >= n_burn_in and i % 5 == 0:
            saved_params.append(params)
            saved_R0ts.append(R0t)

        if i % 4 == 0:
            params_r = np.round(params + [log_prob_new, log_prob_old, log_prob_new - log_prob_old, params[0] / params[3]], 5)
            print(f"iter. {i}=> beta:{params_r[0]}  q:{params_r[1]}  g:{params_r[2]}  gamma:{params_r[3]}  "
                + f"log prob new:{params_r[4]}  log prob old:{params_r[5]}  diff:{params_r[6]}"
                # + f"log prob diff:{params_r[6]}"
                # + f"  R0:{params_r[7]}")#  R0 -2nd week:{params_r[8]}"
                )
            t1 = time.time()
            print("Iter %d: Time %.2f | Runtime: %.2f" % (i, t1 - start_time, t1 - t0))
            print(f"B:\n{np.round(B).astype(int)}")
            t0 = t1

    R0s = [p[0] / p[3] for p in saved_params]

    # 80% CI
    CI_FACTOR = 1.28
    R0_low = np.mean(R0s) - CI_FACTOR * np.std(R0s)
    R0_high = np.mean(R0s) + CI_FACTOR * np.std(R0s)

    R0ts_mean = np.mean(saved_R0ts, axis=0)
    R0ts_std = np.std(saved_R0ts, axis=0)
    R0ts_low = R0ts_mean - CI_FACTOR * R0ts_std
    R0ts_high = R0ts_mean + CI_FACTOR * R0ts_std

    return B, np.mean(saved_params, axis=0), np.std(saved_params, axis=0), (R0_low, R0_high), (R0ts_low, R0ts_high)


def update_params(B, C, D, P, I, S, E, inits, params, prior_params, rand_walk_stds,
                  N, t_end, t_ctrl, epsilon, bounds):
    """
    update beta, q, g, gamma with independent MCMC sampling
    each of B, C, D is a list of binomial distributions. The prior is a gamma distribution for each parameter 
    
    proposal distribution is univariate gaussian centered at previous value and sigma from rand_walk_stds (there
    are four; one for each param). 

    """
    def fn(x):
        """
        here x is equal to one of beta, q, g, gamma. since we compute the same likelihood
        function to update each of the params, it is sufficient to use this generic function
        instead of writing one fn function for each param.

        other_data['which_param'] stores the parameter to update. it is an index of params

        """
        beta, q, g, gamma = x
        
        pC = 1 - np.exp(-g)
        pR = 1 - np.exp(-gamma)
        P = compute_P(transmission_rate(beta, q, t_ctrl, t_end), I, N)

        # log likelihood
        # add epsilon to avoid log 0.
        logB = np.sum(np.log(sp.stats.binom(S, P).pmf(B) + epsilon))
        logC = np.sum(np.log(sp.stats.binom(E, pC).pmf(C) + epsilon))
        logD = np.sum(np.log(sp.stats.binom(I, pR).pmf(D) + epsilon))

        # assert not np.isnan(logB)
        # assert not np.isnan(logC)
        # assert not np.isnan(logD)

        # log prior
        # log_prior = 0
        # for i in range(4):
        #     a, b = prior_params[i]
        #     log_prior += np.log(sp.stats.gamma(a, b).pdf(x[i])+epsilon)
        # assert not np.isnan(log_prior)
        
        return -(logB + logC + logD)
    
    log_prob_old = fn(np.array(params))
    # 'trust-constr'
    params_new = sp.optimize.minimize(fn, x0=np.array(params), method='SLSQP', bounds=bounds).x
    log_prob_new = fn(params_new)
    
    t_rate = transmission_rate(params_new[0], params_new[1], t_ctrl, t_end)
    return params_new.tolist(), compute_P(t_rate, I, N), t_rate / params_new[3], log_prob_new, log_prob_old


def compute_I(i0, t_end, C, D):
    """
    I(0) = i0
    I(t+1) = I(t) + C(t) - D(t) for t >= 0

    can be simplified to I(t+1) = i0+sum(C[:t]-D[:t])
    """
    return i0 + np.concatenate(([0], np.cumsum(C - D)[:-1]))

def transmission_rate(beta, q, t_ctrl, t_end):
    """
    rate of transmission on day t, ie. the number of
    newly infected individuals on day t.
    
    This is defined to be beta prior to t_ctrl and beta*exp(-q(t-t_ctrl)) after t_ctrl

    Note: this is different from R0
    """
    trans_rate = np.ones((t_end, )) * beta
    if t_ctrl < t_end:
        ctrl_indices = np.array(range(t_ctrl, t_end))
        trans_rate[ctrl_indices] = beta * np.exp(-q * (ctrl_indices - t_ctrl))

    assert trans_rate.all() >= 0
    # except AssertionError as e:
    #     print(beta, q, trans_rate[trans_rate < 0])
    #     raise e
    return trans_rate

def compute_P(trans_rate, I, N):
    """
    P[t] = 1 - exp(-BETA[t] * I[t] / N)
    here BETA[t] = time dependent transmission rate
    """
    try:
        return 1 - np.exp(-trans_rate * I / N)
    except:
        print(trans_rate, I)
        raise ValueError


def create_dataset(inits, beta, q, g, gamma, t_ctrl, tau):
    s0, e0, i0 = inits
    N = s0
    S = [s0]
    E = [e0]
    I = [i0]
    R = [N - s0 - e0 - i0]
    B = []
    C = []
    D = []
    P = []
    t_rate = transmission_rate(beta, q, t_ctrl, tau - 1)
    
    t = 0
    while t < tau - 1 and I[t] + E[t] > 0:
        P.append(1-np.exp(-t_rate[t]*I[t]/N))
        pC = 1-np.exp(-g)
        pR = 1-np.exp(-gamma)

        B.append(np.random.binomial(S[t], P[t]))
        C.append(np.random.binomial(E[t], pC))
        D.append(np.random.binomial(I[t], pR))

        S.append(S[t] - B[t])
        E.append(E[t] + B[t] - C[t])
        I.append(I[t] + C[t] - D[t])
        R.append(N - S[t] - E[t] - I[t])
        t += 1

        # print(t, B[t], C[t], D[t], E[t], I[t], P[t])
    if np.sum(B) > 20:
        print(f"number of observations:{t}")
        return np.sum(B), np.array(C), np.array(D)
    else:
        return create_dataset(inits, beta, q, g, gamma, t_ctrl, tau)


def read_dataset(filepath, n=3):
    def moving_average(a) :
        ret = np.cumsum(a, dtype=int)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] // n
    
    df = pd.read_csv(filepath)
    C = moving_average(df.num_confirmed_that_day[10:-1].to_numpy())
    D = moving_average(df.num_death_that_day[10:-1].to_numpy()+df.num_recovered_that_day[10:-1].to_numpy())

    return C, D

if __name__ == '__main__':
    # N = 5364500
    # t_end = 100
    # inits = [N, 1, 0]
    # priors = [(2, 10)]*4
    # rand_walk_stds = [0.01, 0.01, 0.01, 0.01]
    # t_ctrl = 130
    # tau = 1000
    # n_iter = 30000
    # n_burn_in = 3000
    # m, C, D = create_dataset(inits, beta=0.2, q=0.2, g=0.2, gamma=0.1429, t_ctrl=t_ctrl, tau=tau)
    
    N = 60550075#51.57*10**3 # population
    # S(0), E(0), I(0)
    inits = [N, 0, 2]
    priors = [(2, 10)]*4 # no need to change
    rand_walk_stds = [0.003, 0.003, 0.003, 0.003] # no need to change
    t_ctrl = 36          # day on which control measurements were introduced
    tau = 1000           # no need to change
    n_iter = 1000       # no need to change
    n_burn_in = 600    # no need to change
    C, D = read_dataset('../datasets/italy_mar_24.csv', n=3) # k = smoothing factor
    m = np.sum(C)


    incubation_range = [1/8, 1/4]
    infectious_range = [1/8, 1/2]
    
    bounds=[[0., 2], [0, 10], incubation_range, infectious_range]

    print(f"1/mean incubation period: {incubation_range} days, 1/mean infectious period: {infectious_range} days")
    params_mean, params_std, R0_conf, R0ts_conf = train(C, D, N, inits, priors, 
        rand_walk_stds, t_ctrl, tau, n_iter, n_burn_in, m, bounds)[1:]
    print(f"parameters (beta, q, g, gamma): mean: {params_mean}, std={params_std}\n\n"
          +f"R0 80% confidence interval: {R0_conf}\n\n"
          +f"R0[t] 80% confidence interval: {R0ts_conf}"
        )
    low, high = R0ts_conf
    line1, = plt.plot(range(len(low)), low, marker='o', linestyle='solid', linewidth=2, markersize=6, label='lower bound')
    line2, = plt.plot(range(len(low)), high, marker='o', linestyle='solid', linewidth=2, markersize=6, label='upper bound')
 
    plt.xlabel('day t', fontsize=12)
    plt.ylabel('R0_t', fontsize=12)
     
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
     
    plt.legend(handles=[line1, line2], fontsize=12)
    plt.show()
