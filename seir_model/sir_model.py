import argparse
import numpy as np 
import scipy as sp
import pandas as pd
from scipy import stats, optimize, interpolate
import matplotlib.pyplot as plt

import time

"""
The model learns its parameters from C and D. see docstring of train()
These parameters can be used for R0 estimation and for making other 
predictions.

I generated a dummy dataset that was in paper section 3.3. I set 
N=s0=500 instead of 5364500 for speed. see __name__ == __main__:
"""


def metropolis_hastings(x, data, fn, proposal, conditions_fn, burn_in=1):
    """
    get 1 sample from a distribution p(x) ~ k*fn(x) given proposal
    distribution proposal(x) with metropolis hastings algorithm

        * the new sample has to satisfy the conditions in conditions_fn
        * data is a list of additional distribution, variables etc that are
          required to compute the functions
        * assumes proposal distribution is symmetric, ie: q(x'|x) = q(x|x')
        * fn returns log prob. for numeric stability

    returns: one sample from p(x) and corresponding data
    """
    old_log_prob = fn(x, data)
    while burn_in:
        burn_in -= 1
        x_new, data_new = proposal(x, data, conditions_fn)
        accept_log_prob = min(0, fn(x_new, data_new) - fn(x, data))
        if np.random.binomial(1, np.exp(accept_log_prob)):
            # if accept_log_prob < 0: print("accepted new state")
            x, data = x_new, data_new #, fn(x_new, data_new), fn(x, data)
        else:
            pass
    return x, data, fn(x, data), old_log_prob



def train(N, D_wild, inits, params, priors, rand_walk_stds, t_ctrl, tau, n_iter, n_burn_in, bounds, save_freq):
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
    t_end = len(N)
    assert t_end < tau
    assert t_ctrl < tau
    assert len(N) == len(D_wild)
    assert n_burn_in < n_iter

    # initialize model parameters
    i_mild0, i_wild0 = inits
    beta, q, delta, gamma_mild, gamma_wild, k = params
    print("Initializating Variables...")
    S, I_mild, I_wild, C, D_mild, P, t_rate, N = initialize(inits, params, N, D_wild, t_ctrl)
    epsilon = 1e-16
    print("Initialization Complete.")
    check_rep_inv(S, I_mild, I_wild, C, D_mild, D_wild, P)

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
        beta, q, delta, gamma_mild, gamma_wild, k = params
        assert (S==compute_S(C, N, inits)).all()
        assert (I_mild==compute_I(i_mild0, round_int(C*delta), D_mild)).all()
        assert (I_wild==compute_I(i_wild0, C-round_int(C*delta), D_wild)).all()
        assert (P == compute_P(transmission_rate(beta, q, t_ctrl, t_end), I_mild, I_wild, N)).all()

        C, S, I_mild, I_wild, P, _, _ = sample_C(C, [S, I_mild, I_wild, D_mild, D_wild, N, P], 
                                                           inits, params, t_ctrl, epsilon)
        check_rep_inv(S, I_mild, I_wild, C, D_mild, D_wild, P)
        assert (S==compute_S(C, N, inits)).all()
        assert (I_mild==compute_I(i_mild0, round_int(C*delta), D_mild)).all()
        assert (I_wild==compute_I(i_wild0, C-round_int(C*delta), D_wild)).all()
        assert (P == compute_P(transmission_rate(beta, q, t_ctrl, t_end), I_mild, I_wild, N)).all()

        D_mild, I_mild, P, _, _ = sample_D_mild(D_mild, [I_mild, I_wild, C, N], inits, params, t_ctrl, epsilon)
        check_rep_inv(S, I_mild, I_wild, C, D_mild, D_wild, P)
        assert (S==compute_S(C, N, inits)).all()
        assert (I_mild==compute_I(i_mild0, round_int(C*delta), D_mild)).all()
        assert (I_wild==compute_I(i_wild0, C-round_int(C*delta), D_wild)).all()
        assert (P == compute_P(transmission_rate(beta, q, t_ctrl, t_end), I_mild, I_wild, N)).all()

        # MCMC update for params and P
        # I is fixed by C and D and doesn't need to be updated
        params, S, I_mild, I_wild, P, N, R0t, log_prob_new, log_prob_old = sample_params(params, 
                                                                    [S, I_mild, I_wild, C, D_mild, D_wild, P, N], 
                                                                    inits, priors, rand_walk_stds, t_ctrl, epsilon, bounds
                                                                   )
        check_rep_inv(S, I_mild, I_wild, C, D_mild, D_wild, P)
        
        if i >= n_burn_in and i % save_freq == 0:
            saved_params.append(params)
            saved_R0ts.append(R0t)

        if i % 50 == 0:
            beta, q, delta, gamma_mild, gamma_wild, k = np.round(params, 5)
            params_dict = {'beta': beta, 'q': q, 'delta': delta, 
                           'gamma_mild':gamma_mild, 'gamma_wild':gamma_wild, 'k': k,
                           'log_prob_new':np.round(log_prob_new, 5), 'diff':np.round(log_prob_new-log_prob_old, 5) 
                           }
            print(f"iter {i}:\n{params_dict}")
            t1 = time.time()
            print("iter %d: Time %.2f | Runtime: %.2f" % (i, t1 - start_time, t1 - t0))
            print(f"C:\n{C}")
            print(f"D_mild:\n{D_mild}")
            print(f"D_wild:\n{D_wild}")
            t0 = t1

    R0s = [(sum(D_mild)+sum(D_wild)) * p[0] / (sum(D_mild)*p[3]+sum(D_wild)*p[4]) for p in saved_params]

    # 80% CI
    CI_FACTOR = 1.96
    R0_low = np.mean(R0s) - CI_FACTOR * np.std(R0s)
    R0_high = np.mean(R0s) + CI_FACTOR * np.std(R0s)

    R0ts_mean = np.mean(saved_R0ts, axis=0)
    R0ts_std = np.std(saved_R0ts, axis=0)
    R0ts_low = R0ts_mean - CI_FACTOR * R0ts_std
    R0ts_high = R0ts_mean + CI_FACTOR * R0ts_std

    return C, np.mean(saved_params, axis=0), np.std(saved_params, axis=0), (R0_low, R0_high), (R0ts_low, R0ts_high)


def round_int(x):
    return np.floor(x+0.5).astype(int)

def check_rep_inv(S, I_mild, I_wild, C, D_mild, D_wild, P):
    """
    check rep invariant
    """
    assert (I_mild >= 0).all()    
    assert (I_wild >= 0).all()    
    assert (S >= 0).all()
    assert (I_mild + I_wild >= 0).all()
    assert (C >= 0).all()
    assert (D_mild >= 0).all()
    assert (D_wild >= 0).all()
    # P is a list of binomial parameters
    assert (1 >= P).all() and (P >= 0).all()

def sample_x(x, data, conditions_fn, data_fn):
    """
    x:  a sample from p(B|.)
    data = [P, I, S, E], and P doesn't depend on x
    
    sampling for B works as follows (according to paper)
        1. randomly select an index t' such that B[t'] > 0
        2. set B[t'] -= 1
        3. randomly select an index t^
        4. set B[t^] += 1
        5. compute S and E for this new B
        6. Verify that E >= 0 (S >= 0 obviously since sum(B) is constant)
        7. Verify I+E>0
    The authors suggested to select N*10% indices instead of 1 for faster convergence
    """
    n_tries = 0
    x_new = np.copy(x)
    while n_tries < 1000:
        n_tries += 1
        t_new = np.random.choice(np.nonzero(x_new >= 1)[0], min(5, len(np.nonzero(x_new)[0])), replace=False)
        t_tilde = np.random.choice(range(len(x)), len(t_new), replace=False)
        # t_new += 1
        assert(x_new[t_new] >= 1).all()
        one_off = np.random.binomial(1, 0.5)
        if one_off:
            change_add = 1
            change_subs = 1
        else:
            # 80 and 79 makes the dist symmetric
            # 79 is the solution 'y' of
            # (n+n/y)-(n+n/y)/80) = n
            change_add = np.copy(x_new[t_tilde]//99)
            change_subs = np.copy(x_new[t_new]//100)
        
        x_new[t_new] -= change_subs
        x_new[t_tilde] += change_add
        
        data_new = data_fn(x_new)

        if conditions_fn(x_new, data_new):
            # assert (S_new >= x_new).all()
            # assert(x_new >= 0).all()
            return x_new, data_new
        else:
            # revert back the changes
            x_new[t_new] += change_subs
            x_new[t_tilde] -= change_add

    # assert (E >= 0).all() and (E+I > 0).all()
    # print("no sample found")
    return x, data


def sample_C(C, variables, inits, params, t_ctrl, epsilon):
    """
    get a sample from p(B|C, D, params) using metropolis hastings
    """

    def fn(x, data):
        S, I_mild, I_wild, P = data
        # add epsilon to prevent log 0.
        return np.sum(np.log(sp.stats.binom(S, P).pmf(x)+epsilon))


    def proposal(x, data, conditions_fn):
        def data_fn(x):
            S_new = compute_S(x, N, inits)
            I_mild_new = compute_I(i_mild0, round_int(delta*x), D_mild)
            I_wild_new = compute_I(i_wild0, x - round_int(delta*x), D_wild)
            P_new = compute_P(transmission_rate(beta, q, t_ctrl, t_end), I_mild_new, I_wild_new, N)
            return S_new, I_mild_new, I_wild_new, P_new
        return sample_x(x, data, conditions_fn, data_fn)


    def conditions_fn(x, data):
        S, I_mild, I_wild, P = data
        old_S, old_I_mild, old_I_wild = variables[:3]
        # print((S-old_S).astype(int))
        return  (S>=0).all() and (I_mild>=0).all() and (I_wild>=0).all()


    t_end = len(C)
    i_mild0, i_wild0 = inits
    beta, q, delta, gamma_mild, gamma_wild, k = params
    S, I_mild, I_wild, D_mild, D_wild, N, P = variables
    
    data = [S, I_mild, I_wild, P]
    C, data, log_prob_new, log_prob_old = metropolis_hastings(C, data, fn, proposal, conditions_fn, burn_in=1)
    S, I_mild, I_wild, P = data

    return C, S, I_mild, I_wild, P, log_prob_new, log_prob_old


def sample_D_mild(D_mild, variables, inits, params, t_ctrl, epsilon):
    """
    get a sample from p(B|C, D, params) using metropolis hastings
    """

    def fn(x, data):
        I_mild = data[0]
        # assert (S >= x).all()
        # assert (x >= 0).all()
        # add epsilon to prevent log 0.
        pR = 1-np.exp(-gamma_mild)
        assert 0 <= pR <= 1
        assert not np.isnan(pR)
        return np.sum(np.log(sp.stats.binom(I_mild, pR).pmf(x)+epsilon))
        

    def proposal(x, data, conditions_fn):
        I_mild = data[0]
        def data_fn(x):
            I_mild_new = compute_I(i_mild0, round_int(C*delta), x)
            return [I_mild_new]
        
        return sample_x(x, data, conditions_fn, data_fn)

    def conditions_fn(x, data):
        I_mild = data[0]
        return (I_mild>=0).all()

    t_end = len(D_mild)
    i_mild0, i_wild0 = inits
    beta, q, delta, gamma_mild, gamma_wild, k = params
    I_mild, I_wild, C, N = variables
    data = [I_mild]
    D_mild, data, log_prob_new, log_prob_old = metropolis_hastings(D_mild, data, fn, proposal, conditions_fn, burn_in=1)
    I_mild = data[0]
    P = compute_P(transmission_rate(beta, q, t_ctrl, t_end), I_mild, I_wild, N)
    return [D_mild] + data + [P, log_prob_new, log_prob_old]


def sample_params(params, variables, inits, priors, rand_walk_stds, t_ctrl, epsilon, bounds):
    """
    update beta, q, g, gamma with independent MCMC sampling
    each of B, C, D is a list of binomial distributions. The prior is a gamma distribution for each parameter 
    
    proposal distribution is univariate gaussian centered at previous value and sigma from rand_walk_stds (there
    are four; one for each param). 

    """
    def fn(x, data):
        """
        here x is equal to one of beta, q, g, gamma. since we compute the same likelihood
        function to update each of the params, it is sufficient to use this generic function
        instead of writing one fn function for each param.

        other_data['which_param'] stores the parameter to update. it is an index of params

        """
        beta, q, delta, gamma_mild, gamma_wild, k = x
        S, I_mild, I_wild, P, N = data

        pR_mild = 1 - np.exp(-gamma_mild)
        pR_wild = 1 - np.exp(-gamma_wild)

        # log likelihood
        # add epsilon to avoid log 0.
        logC = np.sum(np.log(sp.stats.binom(S, P).pmf(C) + epsilon))

        logD_mild = np.sum(np.log(sp.stats.binom(I_mild, pR_mild).pmf(D_mild) + epsilon))
        logD_wild = np.sum(np.log(sp.stats.binom(I_wild, pR_wild).pmf(D_wild) + epsilon))

        assert not np.isnan(logC)
        assert not np.isnan(logD_mild)
        assert not np.isnan(logD_wild)

        # log prior
        log_prior = 0
        for i in range(len(priors)):
            a, b = priors[i]
            log_prior += np.log(sp.stats.gamma(a, b).pdf(x[i])+epsilon)
        assert not np.isnan(log_prior)        
        return logC + logD_mild + logD_wild + log_prior

    def proposal(x, data, conditions_fn):
        """
        see docstring for previous function
        """
        S, I_mild, I_wild, P, N = data
        n_tries = 0
        while n_tries < 1000:
            n_tries += 1
            
            x_new = np.random.normal(x, rand_walk_stds)
            beta, q, delta, gamma_mild, gamma_wild, k = x_new
            
            N_new = round_int(N*old_k/k)
            N_new[N_new<1] = 1
            S_new = compute_S(C, N_new, inits)
            I_mild_new =compute_I(i_mild0, round_int(C*delta), D_mild)
            I_wild_new =compute_I(i_wild0, C-round_int(C*delta), D_wild)            
            P_new = compute_P(transmission_rate(beta, q, t_ctrl, t_end), I_mild_new, I_wild_new, N_new)
            data_new = [S_new, I_mild_new, I_wild_new, P_new, N_new]

            if conditions_fn(x_new, data_new):
                # print(x_new-x, fn(x_new, data_new)-fn(x, data))
                return x_new, data_new
        # print("sample not found")
        return x, data
    
    def conditions_fn(x, data):
        """
        all parameters should be non-negative
        """
        beta, q, delta, gamma_mild, gamma_wild, k = x
        S, I_mild, I_wild, P, N = data
        
        if not (x > 0).all():
            return False
        
        if not (S >= 0).all() or not (I_mild >= 0).all() or not (I_wild >= 0).all():
            return False

        for i in range(len(bounds)):
            a, b = bounds[i]
            param = x[i]
            if x[i] < a or x[i] > b:
                return False
        return True

    i_mild0, i_wild0 = inits
    S, I_mild, I_wild, C, D_mild, D_wild, P, N = variables
    t_end = len(N)
    beta, q, delta, gamma_mild, gamma_wild, k = params
    old_k = k    
    data = [S, I_mild, I_wild, P, N]

    params_new, data, log_prob_new, log_prob_old = metropolis_hastings(np.array(params), data, fn, proposal, conditions_fn, burn_in=1)
    beta, q, delta, gamma_mild, gamma_wild, k = params_new
    t_rate = transmission_rate(beta, q, t_ctrl, t_end)
    # R0t = (sum(D_mild)+sum(D_wild))*t_rate /((sum(D_mild)*gamma_mild+sum(D_wild)*gamma_wild)) * S/N
    R0t = t_rate /(delta*gamma_mild+(1-delta)*gamma_wild) * S/N
    
    S, I_mild, I_wild, P, N = data
    
    return params_new.tolist(), S, I_mild, I_wild, P, N, R0t, log_prob_new, log_prob_old


def compute_S(C, N, inits):
    """
    S(0) = s0
    S(t+1) = S(t) - B(t) + N(t+1)-N(t) for t >= 0

    can be simplified to S(t+1) = s0 - sum(B[:t])
    """
    imild0, iwild0 = inits
    return N[0] - imild0 - iwild0 - np.concatenate(([0], np.cumsum(C)[:-1])) + N - N[0]


def compute_I(i0, C, D):
    """
    computes either I_mild or I_wild depending on the inputs
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
    return trans_rate

def compute_P(trans_rate, I_mild, I_wild, N):
    """
    P[t] = 1 - exp(-BETA[t] * I[t] / N)
    here BETA[t] = time dependent transmission rate
    """
    P = 1 - np.exp(-trans_rate * (I_mild+I_wild) / N)
    return P


def compute_rand_walk_cov(t, t_skip, C0, C_t, mean_t, mean_tm1, x_t, epsilon):
    assert t_skip > 2
    if t < t_skip:
        return C0
    else:
        return (t-1)/t * C_t + 2.4**2/len(C0) * (t*mean_tm1@mean_tm1.T-(t+1)*mean_t@mean_t.T + x_t@x_t.T+epsilon*np.identity(len(C0)))


def read_dataset(filepath, n=3, offset=1, last_offset=1):
    def moving_average(a) :
        ret = np.cumsum(a, dtype=int)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] // n
    
    df = pd.read_csv(filepath)
    N = moving_average(df.num_confirmed[offset:-last_offset].to_numpy())
    D_wild = moving_average(df.num_confirmed_that_day[offset:-last_offset].to_numpy())
    
    N[N < 1] = 1
    D_wild[D_wild <= 0] = 0
    return round_int(N), round_int(D_wild)


def initialize(inits, params, N, D_wild, t_ctrl):
    beta, q, delta, gamma_mild, gamma_wild, k = params
    i_mild0, i_wild0 = inits
    P, C, D_mild, N_new = [], [], [], []
    S, I_mild, I_wild = [N[0]-i_mild0-i_wild0], [i_mild0], [i_wild0] 
    t_rate = transmission_rate(beta, q, t_ctrl, len(N))
    
    for t in range(len(N)-1):
        s, i_mild, i_wild = S[t], I_mild[t], I_wild[t]
        # print(i_mild, i_wild)
        p = 1-np.exp(-t_rate[t]*(i_mild+i_wild)/N[t])
        assert 0 <= p <=1
        
        d_wild = int(D_wild[t])
        c = round_int(s*p)
        d_mild = round_int(i_mild*(1-np.exp(-gamma_mild)))

        c_mild = round_int(c*delta)
        c_wild = c-c_mild
        
        s = s - c + N[t+1] - N[t]
        i_mild = i_mild + c_mild - d_mild
        i_wild = i_wild + c_wild - d_wild
        print(f"t: {t}, D_mild[t]: {d_mild}, D_wild[t]: {d_wild}, I_wild[t]: {i_wild}")
        assert i_wild >= 0
        S.append(s)
        I_mild.append(i_mild)
        I_wild.append(i_wild)
        
        C.append(c)
        D_mild.append(d_mild)
        P.append(p)

    # last step
    p = 1-np.exp(-t_rate[-1]*(I_mild[-1]+I_wild[-1])/N[-1])
    c = np.random.binomial(S[-1], p)
    d_mild = int(I_mild[-1]*(1-np.exp(-gamma_mild)))

    C.append(c)
    P.append(p)
    D_mild.append(d_mild)

    return [np.array(S), np.array(I_mild), np.array(I_wild), 
            np.array(C), np.array(D_mild), np.array(P), t_rate, np.array(N)]


if __name__ == '__main__':
    # korea
    # params = [2, 0.05, 0.6, 0.15, 0.33, 0.2]
    # n = 3
    # offset, last_offset = 30, 1
    # lockdown = 37 # 
    # rand_walk_stds = [0.01, 0.002, 0.002, 0.002, 0.002, 0.002] # [0.01, 0.001, 0.001, 0.001, 0.001, 0.001]

    # italy
    
    # wuhan
    # params = [0.5, 0.001, 0.8, 0.18, 0.33, 0.1]
    # n = 5
    # offset, last_offset = 2, 12
    # lockdown = 3

    # new york
    # params = [0.6, 0.001, 0.8, 0.18, 0.33, 0.18]
    # n = 5
    # offset, last_offset = 40, 1
    # lockdown = 50

    # germany
    # params = [0.7, 0.001, 0.6, 0.18, 0.33, 0.22]
    # n = 5
    # offset, last_offset = 33, 1
    # lockdown = 58

    # california
    # params = [0.7, 0.001, 0.7, 0.18, 0.33, 0.22]
    # n = 5
    # offset, last_offset = 43, 1
    # lockdown = 57

    # turkey
    # filename = os.path.join(dirname, '../datasets/china_mar_30.csv')
    # out_filename = os.path.join(dirname, '../output_china_start_jan23_lockdown_none.txt')
    # params = [0.7, 0.001, 0.8, 0.18, 0.33, 0.22]
    # n = 5
    # offset, last_offset = 49, 1
    # lockdown = 66

    # china
    # filename = os.path.join(dirname, '../datasets/china_mar_30.csv')
    # out_filename = os.path.join(dirname, '../output_china_start_jan23_lockdown_jan28.txt')
    # params = [0.8, 0.001, 0.8, 0.18, 0.33, 0.1]
    # n = 5
    # offset, last_offset = 1, 1
    # lockdown = 6
    # rand_walk_stds = [0.008, 0.0005, 0.0005, 0.001, 0.001, 0.0007] # no need to change

    # us
    # filename = os.path.join(dirname, '../datasets/us_mar_30.csv')
    # out_filename = os.path.join(dirname, '../output_us_start_feb25_lockdown_none.txt')
    # params = [0.8, 0.001, 0.8, 0.18, 0.33, 0.1]
    # n = 5
    # offset, last_offset = 34, 1
    # lockdown = 66
    # rand_walk_stds = [0.008, 0.001, 0.001, 0.001, 0.001, 0.001] # no need to change

    import os
    dirname = os.path.dirname(__file__)
    default_in_filename = '../datasets/korea_mar_30.csv'
    default_out_filename = '../output_korea_start_feb19_lockdown_feb26trial4.txt'

    parser = argparse.ArgumentParser(description='Learn an SEIR model for the COVID-19 infected data.')
    parser.add_argument('--infile', type=str, help='Directory for the location of the input file',
                        default=default_in_filename, nargs='?')
    parser.add_argument('--outfile', type=str, help='Directory for the location of the input file',
                        default=default_out_filename, nargs='?')
    # beta, q, delta, gamma_mild, gamma_wild, k, kctrl
    parser.add_argument('--params', type=str, default="0.5, 0.01, 0.5, 0.18, 0.3, 0.1", nargs='?', 
                        help="inits for beta, q, delta, gamma_mild, gamma_wild, k")
    parser.add_argument('--inits', type=str, default="1000, 1000", nargs='?', help="initial values for imild0 and iwild0")
    parser.add_argument('--n', type=int, default=3, nargs='?', help="number of entries to take rolling mean over")
    parser.add_argument('--offset', type=int, default=30, nargs='?', 
                        help="number of days >=1 to exclude in the beginning of the dataset")
    parser.add_argument('--last_offset', type=int, default=1, nargs='?', help="number of days >=1 to exclude at the end of dataset")
    parser.add_argument('--lockdown', type=int, default=37, nargs='?', help="day on which lock down was imposed")
    parser.add_argument('--n_iter', type=int, default=12000, nargs='?', help="number of iterations")
    parser.add_argument('--n_burn_in', type=int, default=5000, nargs='?', help="burn in period for MCMC")
    parser.add_argument('--save_freq', type=int, default=200, nargs='?', help="how often to save samples after burn in")
    parser.add_argument('--rand_walk_stds', type=str, default="0.005, 0.001, 0.005, 0.001, 0.001, 0.002", nargs='?', 
                       help="stds for gaussian random walk in MCMC (one for each param)")

    # beta, q, delta, gamma_mild, gamma_wild, k, kctrl
    bounds=[(0, 2), (0, np.inf), (0.08, 0.92), (0.05, 0.5), (0.05, 0.5), (0, 1)]
    args = parser.parse_args()
    params = [float(param) for param in args.params.split(',')] # italy
    rand_walk_stds = [float(std) for std in args.rand_walk_stds.split(',')]
    assert len(params) == 6 and len(rand_walk_stds) == 6, "Need all parameters and their random walk stds"
    n = args.n
    offset, last_offset = args.offset, args.last_offset
    lockdown = args.lockdown
    in_filename = os.path.join(dirname, args.infile)
    out_filename = os.path.join(dirname, args.outfile)

    t_ctrl = lockdown-offset          # day on which control measurements were introduced
    assert t_ctrl >= 0

    N, D_wild = read_dataset(in_filename, n, offset, last_offset) # k = smoothing factor
    N = round_int(N/params[5])
    N[N < 1] = 1

    # Imild(0), Iwild(0)
    delta = params[2]
    inits = [int(init) for init in args.inits.split(',')]
    priors = [(2, 10)]*len(params) # no need to change
    
    tau = 1000           # no need to change
    n_iter = args.n_iter      # no need to change
    n_burn_in = args.n_burn_in    # no need to change
    save_freq = args.save_freq
    
    
    params_mean, params_std, R0_conf, R0ts_conf = train(N, D_wild, inits, params, priors, 
                                                        rand_walk_stds, t_ctrl, tau, n_iter, n_burn_in, bounds, save_freq
                                                       )[1:]
    print(f"\nFINAL RESULTS\n\ninput file{in_filename}")
    print(f"ouput file: {out_filename}")
    print(f"param inits: {params}")
    param_names = ["beta", "q", "delta", "gamma_mild", "gamma_wild", "k"]
    print(f"parameters mean: {dict(zip(param_names, params_mean))}\nparameters std={dict(zip(param_names, params_std))}\n\n"
          +f"R0 95% confidence interval: {R0_conf}\n\n"
          +f"R0[t] 95% confidence interval: {R0ts_conf}"
        )
    low, high = R0ts_conf
    
    with open(out_filename, 'w') as out:
        out.write("SEIR MODEL FOR R0t PREDICTION\n---   ---   ---   ---   ---\n"
                 +f"dataset name: {in_filename}\n"
                 +f"output filename: {out_filename}\n\n"
                 +f"inits (imild0, iwild0): {inits}, rand_walk_stds:{rand_walk_stds}\n"
                 +f"t_ctrl:{t_ctrl}, t_end:{len(N)}, n_iter:{n_iter}, n_burn_in:{n_burn_in}, save_freq:{save_freq}\n"
                 +f"offset:{offset}, last_offset:{last_offset}, smoothing:{n}\n"
                 +f"bounds:{bounds}\n"
                 +f"param inits:{params}\n"
                 +f"parameters (beta, q, delta, rho, gamma_mild, gamma_wild, k, kctrl): mean: {params_mean}, std={params_std}\n\n"
                 +f"R0 95% confidence interval: {R0_conf}\n\n"
                 +f"R0[t] 95% confidence interval: {R0ts_conf}\n"
                 )
    out.close()

    mean = (low+high)/2
    line1, = plt.plot(range(len(low)), low, marker='.', linestyle='dashed', linewidth=2, markersize=4, label='lower bound')
    line2, = plt.plot(range(len(high)), high, marker='.', linestyle='dashed', linewidth=2, markersize=4, label='upper bound')
    line3, = plt.plot(range(len(high)), mean, marker='o', linestyle='solid', linewidth=2, markersize=5, label='mean')
 
    plt.xlabel('day t', fontsize=12)
    plt.ylabel('R0_t', fontsize=12)
     
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.ylim(0, 10)
    plt.legend(handles=[line1, line2, line3], fontsize=12)
    plt.show()
