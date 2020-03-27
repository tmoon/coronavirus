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



def train(N, D_wild, inits, params, priors, rand_walk_stds, t_ctrl, tau, n_iter, n_burn_in, bounds):
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
    s0, e0, i_mild0, i_wild0 = inits
    beta, q, g_mild, g_wild, gamma_mild, gamma_wild, k = params
    print("Initializating Variables...")
    S, E, I_mild, I_wild, B, C_mild, C_wild, D_mild, P, t_rate = initialize(inits, params, N, D_wild, t_ctrl)
    epsilon = 1e-16
    print("Initialization Complete.")
    check_rep_inv(S, E, I_mild, I_wild, B, C_mild, C_wild, D_mild, D_wild, P)

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
        B, S, E, log_prob_new, log_prob_old = sample_B(B, [P, S, E, I_mild, I_wild, N, C_mild, C_wild], inits, params, t_ctrl, epsilon)
        check_rep_inv(S, E, I_mild, I_wild, B, C_mild, C_wild, D_mild, D_wild, P)

        C_mild, C_wild, E, I_mild, I_wild, _, _ = sample_C(C_mild, C_wild, [B, E, I_mild, I_wild, D_mild, D_wild], 
                                                           inits, params, t_ctrl, epsilon)
        check_rep_inv(S, E, I_mild, I_wild, B, C_mild, C_wild, D_mild, D_wild, P)

        D_mild, I_mild, _, _ = sample_D_mild(D_mild, [E, I_mild, I_wild, C_mild], inits, params, t_ctrl, epsilon)
        check_rep_inv(S, E, I_mild, I_wild, B, C_mild, C_wild, D_mild, D_wild, P)

        # MCMC update for params and P
        # I is fixed by C and D and doesn't need to be updated
        params, P, R0t, log_prob_new, log_prob_old = sample_params(params, [S, E, I_mild, I_wild, B, C_mild, C_wild, D_mild, D_wild, P, N], 
                                                                    inits, priors, rand_walk_stds, t_ctrl, epsilon, bounds
                                                                   )
        check_rep_inv(S, E, I_mild, I_wild, B, C_mild, C_wild, D_mild, D_wild, P)
        
        if i >= n_burn_in and i % 10 == 0:
            saved_params.append(params)
            saved_R0ts.append(R0t)

        if i % 2 == 0:
            beta, q, g_mild, g_wild, gamma_mild, gamma_wild, k = params
            params_dict = {'beta': beta, 'q': q, 'g_mild': g_mild, 'g_wild':g_wild, 
                           'gamma_mild':gamma_mild, 'gamma_wild':gamma_wild, 'k': k,
                           'log_prob_new':log_prob_new, 'diff':log_prob_new-log_prob_old, 
                           }
            print(f"iter. {i}=> {params_dict}")
            t1 = time.time()
            print("Iter %d: Time %.2f | Runtime: %.2f" % (i, t1 - start_time, t1 - t0))
            print(f"B:\n{B}")
            t0 = t1

    R0s = [(sum(D_mild)+sum(D_wild)) * p[0] / (sum(D_mild)*p[3]+sum(D_wild)*p[4]) for p in saved_params]

    # 80% CI
    CI_FACTOR = 1.28
    R0_low = np.mean(R0s) - CI_FACTOR * np.std(R0s)
    R0_high = np.mean(R0s) + CI_FACTOR * np.std(R0s)

    R0ts_mean = np.mean(saved_R0ts, axis=0)
    R0ts_std = np.std(saved_R0ts, axis=0)
    R0ts_low = R0ts_mean - CI_FACTOR * R0ts_std
    R0ts_high = R0ts_mean + CI_FACTOR * R0ts_std

    return B, np.mean(saved_params, axis=0), np.std(saved_params, axis=0), (R0_low, R0_high), (R0ts_low, R0ts_high)


def check_rep_inv(S, E, I_mild, I_wild, B, C_mild, C_wild, D_mild, D_wild, P):
    """
    check rep invariant
    """
    assert (I_mild >= 0).all()    
    assert (I_wild >= 0).all()    
    assert (S >= 0).all()
    assert (E >= 0).all()
    assert (E + I_mild + I_wild > 0).all()
    assert (B >= 0).all()
    assert (C_mild >= 0).all()
    assert (C_wild >= 0).all()
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
        t_new = np.random.choice(np.nonzero(x_new)[0], min(20, len(np.nonzero(x_new)[0])), replace=False)
        t_tilde = np.random.choice(range(len(x)), len(t_new), replace=False)
        
        x_new[t_new] -= 1
        x_new[t_tilde] += 1
        data_new = data_fn(x_new)

        if conditions_fn(x_new, data_new):
            # assert (S_new >= x_new).all()
            # assert(x_new >= 0).all()
            return x_new, data_new
        else:
            # revert back the changes
            x_new[t_new] += 1
            x_new[t_tilde] -= 1
    # assert (E >= 0).all() and (E+I > 0).all()
    return x, data


def sample_B(B, variables, inits, params, t_ctrl, epsilon):
    """
    get a sample from p(B|C, D, params) using metropolis hastings
    """

    def fn(x, data):
        S, E = data
        # assert (S >= x).all()
        # assert (x >= 0).all()
        # add epsilon to prevent log 0.
        return np.sum(np.log(sp.stats.binom(S, P).pmf(x)+epsilon))
        

    def proposal(x, data, conditions_fn):
        S, E = data
        def data_fn(x):
            S_new = compute_S(s0, x, N)
            E_new = compute_E(e0, x, C_mild, C_wild)
            return S_new, E_new

        return sample_x(x, data, conditions_fn, data_fn)

    def conditions_fn(x, data):
        S, E = data
        return np.sum(x) == sum_B and (E>=0).all() and (E+I_mild+I_wild>0).all()

    s0, e0, i_mild0, i_wild0 = inits
    P, S, E, I_mild, I_wild, N, C_mild, C_wild = variables
    data = [S, E]
    sum_B = np.sum(B)
    B, data, log_prob_new, log_prob_old = metropolis_hastings(B, data, fn, proposal, conditions_fn, burn_in=30)
    return [B] + data + [log_prob_new, log_prob_old]


def sample_C(C_mild, C_wild, variables, inits, params, t_ctrl, epsilon):
    """
    get a sample from p(B|C, D, params) using metropolis hastings
    """

    def fn_mild(x, data):
        E, I_mild = data
        # add epsilon to prevent log 0.
        pC = 1-np.exp(-g_mild)
        assert 0 <= pC <= 1
        assert not np.isnan(pC)
        return np.sum(np.log(sp.stats.binom(E, pC).pmf(x)+epsilon))

    def fn_wild(x, data):
        E, I_wild = data
        # add epsilon to prevent log 0.
        pC = 1-np.exp(-g_wild)
        assert 0 <= pC <= 1
        assert not np.isnan(pC)
        return np.sum(np.log(sp.stats.binom(E, pC).pmf(x)+epsilon))
        

    def proposal_mild(x, data, conditions_fn):
        E, I_mild = data
        def data_fn(x):
            E_new = compute_E(e0, B, x, C_wild)
            I_mild_new = compute_I(i_mild0, x, D_mild)
            return E_new, I_mild_new
        return sample_x(x, data, conditions_fn, data_fn)


    def proposal_wild(x, data, conditions_fn):
        E, I_wild = data
        def data_fn(x):
            E_new = compute_E(e0, B, C_mild, x)
            I_wild_new = compute_I(i_wild0, x, D_wild)
            return E_new, I_wild_new
        return sample_x(x, data, conditions_fn, data_fn)


    def conditions_fn_mild(x, data):
        E, I_mild = data
        return np.sum(x) == sum_C_mild and (E>=0).all() and (E+I>0).all()

    def conditions_fn_wild(x, data):
        E, I_wild = data
        return np.sum(x) == sum_C_wild and (E>=0).all() and (E+I_mild+I_wild>0).all()


    s0, e0, i_mild0, i_wild0 = inits
    beta, q, g_mild, g_wild, gamma_mild, gamma_wild, k = params
    B, E, I_mild, I_wild, D_mild, D_wild = variables
    
    data_mild = [E, I_mild]
    sum_C_mild = np.sum(C_mild)
    C_mild, data_mild, log_prob_new, log_prob_old = metropolis_hastings(C_mild, data_mild, fn_mild, proposal_mild, 
                                                                        conditions_fn_mild, burn_in=30)
    E, I_mild = data_mild
    
    data_wild = [E, I_wild]
    sum_C_wild = np.sum(C_wild)
    C_wild, data_wild, log_prob_new, log_prob_old = metropolis_hastings(C_wild, data_wild, fn_wild, proposal_wild, 
                                                                        conditions_fn_wild, burn_in=30)
    E, I_wild = data_wild

    return C_mild, C_wild, E, I_mild, I_wild, log_prob_new, log_prob_old


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
            I_mild_new = compute_I(i_mild0, C_mild, x)
            return [I_mild_new]
        
        return sample_x(x, data, conditions_fn, data_fn)

    def conditions_fn(x, data):
        I_mild = data[0]
        return np.sum(x) == sum_D_mild and (E>=0).all() and (E+I_mild+I_wild>0).all()

    s0, e0, i_mild0, i_wild0 = inits
    beta, q, g_mild, g_wild, gamma_mild, gamma_wild, k = params
    E, I_mild, I_wild, C_mild = variables
    data = [I_mild]
    sum_D_mild = np.sum(D_mild)
    D_mild, data, log_prob_new, log_prob_old = metropolis_hastings(D_mild, data, fn, proposal, conditions_fn, burn_in=30)
    return [D_mild] + data + [log_prob_new, log_prob_old]


def sample_params(params, variables, inits, priors, rand_walk_stds, t_ctrl, epsilon, bounds):
    """
    update beta, q, g, gamma with independent MCMC sampling
    each of B, C, D is a list of binomial distributions. The prior is a gamma distribution for each parameter 
    
    proposal distribution is univariate gaussian centered at previous value and sigma from rand_walk_stds (there
    are four; one for each param). 

    """
    def fn(x, data=None):
        """
        here x is equal to one of beta, q, g, gamma. since we compute the same likelihood
        function to update each of the params, it is sufficient to use this generic function
        instead of writing one fn function for each param.

        other_data['which_param'] stores the parameter to update. it is an index of params

        """
        beta, q, g_mild, g_wild, gamma_mild, gamma_wild, k = x
        
        pC_mild = 1 - np.exp(-g_mild)
        pC_wild = 1 - np.exp(-g_wild)
        pR_mild = 1 - np.exp(-gamma_mild)
        pR_wild = 1 - np.exp(-gamma_wild)
        P = compute_P(transmission_rate(beta, q, t_ctrl, t_end), I_mild, I_wild, N)

        # log likelihood
        # add epsilon to avoid log 0.
        logB = np.sum(np.log(sp.stats.binom(S, P).pmf(B) + epsilon))
        
        logC_mild = np.sum(np.log(sp.stats.binom(E, pC_mild).pmf(C_mild) + epsilon))
        logC_wild = np.sum(np.log(sp.stats.binom(E, pC_wild).pmf(C_wild) + epsilon))
        
        logD_mild = np.sum(np.log(sp.stats.binom(I_mild, pR_mild).pmf(D_mild) + epsilon))
        logD_wild = np.sum(np.log(sp.stats.binom(I_wild, pR_wild).pmf(D_wild) + epsilon))

        assert not np.isnan(logB)
        assert not np.isnan(logC_mild)
        assert not np.isnan(logC_wild)
        assert not np.isnan(logD_mild)
        assert not np.isnan(logD_wild)

        # log prior
        log_prior = 0
        for i in range(len(data)):
            a, b = priors[i]
            log_prior += np.log(sp.stats.gamma(a, b).pdf(x[i])+epsilon)
        assert not np.isnan(log_prior)        
        return logB + logC_mild + logC_wild + logD_mild + logD_wild + log_prior

    def proposal(x, data, conditions_fn):
        """
        see docstring for previous function
        """
        n_tries = 0
        while n_tries < 1000:
            n_tries += 1
            x_new = np.random.normal(x, rand_walk_stds)
            if conditions_fn(x_new, data):
                return x_new, data
        # print("sample not found")
        return x, data
    
    def conditions_fn(x, data):
        """
        all parameters should be non-negative
        """
        if not (x > 0).all():
            return False

        for i in range(len(bounds)):
            a, b = bounds[i]
            param = x[i]
            if x < a or x > b:
                return False
        return True

    
    params_new, _, log_prob_new, log_prob_old = metropolis_hastings(np.array(params), None, fn, proposal, conditions_fn)
    beta, q, g_mild, g_wild, gamma_mild, gamma_wild, k = params_new
    t_rate = transmission_rate(beta, q, t_ctrl, t_end)
    R0t = (sum(D_mild)+sum(D_wild))*t_rate /((sum(D_mild)*gamma_mild+sum(D_wild)*gamma_wild))
    return params_new.tolist(), compute_P(t_rate, I_mild, I_wild, N), R0t, log_prob_new, log_prob_old


def compute_S(s0, B, N):
    """
    S(0) = s0
    S(t+1) = S(t) - B(t) + N(t+1)-N(t) for t >= 0

    can be simplified to S(t+1) = s0 - sum(B[:t])
    """
    return s0 - np.concatenate(([0], np.cumsum(B)[:-1])) + N - N[0]


def compute_E(e0, B, C_mild, C_wild):
    """
    E(0) = e0
    E(t+1) = E(t) + B(t) - C_mild(t) - C_wild(t) for t >= 0

    can be simplified to E(t+1) = e0+sum(B[:t]-C_mild[:t]-C_wild[:t])
    """
    return e0 + np.concatenate(([0], np.cumsum(B - C_mild - C_wild)[:-1]))


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
    # except AssertionError as e:
    #     print(beta, q, trans_rate[trans_rate < 0])
    #     raise e
    return trans_rate

def compute_P(trans_rate, I_mild, I_wild, N):
    """
    P[t] = 1 - exp(-BETA[t] * I[t] / N)
    here BETA[t] = time dependent transmission rate
    """
    P = 1 - np.exp(-trans_rate * (I_mild+I_wild) / np.cumsum(N))
    assert (P >= 0).all() and (P <= 1).all()
    return P


def read_dataset(filepath, n=3):
    def moving_average(a) :
        ret = np.cumsum(a, dtype=int)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] // n
    
    df = pd.read_csv(filepath)
    D_wild = moving_average(df.num_confirmed[:-1].to_numpy())
    D_wild[D_wild <= 0] = 0
    
    N = 1+np.copy(D_wild)

    return np.floor(N+0.5).astype(int), np.floor(D_wild+0.5).astype(int)


def initialize(inits, params, N, D_wild, t_ctrl, attempt=100):
    beta, q, g_mild, g_wild, gamma_mild, gamma_wild, k = params
    s0, e0, i_mild0, i_wild0 = inits
    B, P, C_mild, C_wild, D_mild = [], [], [], [], []
    S, E, I_mild, I_wild = [s0], [e0], [i_mild0], [i_wild0] 
    t_rate = transmission_rate(beta, q, t_ctrl, len(N))
    
    for t in range(len(N)):
        s, e, i_mild, i_wild = S[t], E[t], I_mild[t], I_wild[t]
        p = 1-np.exp(-t_rate[t]*(i_mild+i_wild)/N[t])
        assert 0 <= p <=1
        
        b = np.random.binomial(s, p)
        c_mild = np.random.binomial(e, 1-np.exp(-g_mild))
        c_wild = np.random.binomial(e, 1-np.exp(-g_wild))
        d_mild = np.random.binomial(i_mild, 1-np.exp(-gamma_mild))
        d_wild = D_wild[t]

        if t+1 < len(N):
            s = s - b + N[t+1] - N[t]
            e = e + b - c_mild - c_wild
            i_mild = i_mild + c_mild - d_mild
            i_wild = i_wild + c_wild - d_wild
            n_iter = 0
            while not (np.array([p, b, c_mild, c_wild, d_mild, s, e, i_mild, i_wild]) >= 0).all() or e+i_mild+i_wild <= 0:
                n_iter += 1
                if b < 0:
                    s += 1
                    b += 1
                if s < 0:
                    b -= 1
                    e += 1
                if e < 0:
                    e += 1
                    b += 1
                    s -= 1
                if c_mild < 0:
                    c_mild += 1
                    e -= 1
                    i_mild += 1
                if c_wild < 0:
                    c_wild += 1
                    e -= 1
                    i_wild += 1
                
                if d_mild < 1:
                    d_mild += 1
                    c_mild += 1
                    e += 1
                    s += 1
                if i_mild < 1:
                    i_mild += 1
                    c_mild += 1
                    e -= 1

                if i_wild < 1:
                    i_wild += 1
                    c_wild += 1
                    e -= 1
                if p < 0:
                    p = 0.2

                # if attempt <= 0:
                # raise ValueError("could not initialize with given parameters. try different values...")
                # else:
                    # return initialize(inits, params, N, D_wild, t_ctrl, attempt-1)
            print(t, [p, b, c_mild, c_wild, d_mild, d_wild, s, e, i_mild, i_wild])
            S.append(s)
            E.append(e)
            I_mild.append(i_mild)
            I_wild.append(i_wild)
        
        B.append(b)
        C_mild.append(c_mild)
        C_wild.append(c_wild)
        D_mild.append(d_mild)
        P.append(p)

    return [np.array(S), np.array(E), np.array(I_mild), np.array(I_wild),
           np.array(B), np.array(C_mild), np.array(C_wild), np.array(D_mild), 
           np.array(P), t_rate]


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
    
    # S(0), E(0), I(0)
    inits = [100, 1, 1, 1]
    priors = [(2, 10)]*7 # no need to change
    rand_walk_stds = [0.00005]*7 # no need to change
    t_ctrl = 46          # day on which control measurements were introduced
    tau = 1000           # no need to change
    n_iter = 25      # no need to change
    n_burn_in = 15    # no need to change
    N, D_wild = read_dataset('../datasets/italy_mar_24.csv', n=3) # k = smoothing factor
    bounds=[(0, np.inf)]*len(priors)
    params = [10, 0.001, .99, 0.95, 0.99, 0.95, 10]
    N *= params[6]

    
    params_mean, params_std, R0_conf, R0ts_conf = train(N, D_wild, inits, params, priors, 
                                                        rand_walk_stds, t_ctrl, tau, n_iter, n_burn_in, bounds
                                                       )[1:]
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
