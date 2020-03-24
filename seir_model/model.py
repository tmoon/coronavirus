import numpy as np 
import scipy as sp
from scipy import stats, optimize, interpolate
import matplotlib.pyplot as plt

"""
The model learns its parameters from C and D. see docstring of train()
These parameters can be used for R0 estimation and for making other 
predictions.

I generated a dummy dataset that was in paper section 3.3. I set 
N=s0=500 instead of 5364500 for speed. see __name__ == __main__:

The model should be able to estimate the parameters, but it's stuck in 
the sampling step for unobserved variable B.

main issue: the MCMC in update_data() is not initializing to
            any non-zero value.

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
        accept_log_prob = min(0, fn(x_new, data_new)-fn(x, data))
        if np.random.binomial(1, np.exp(accept_log_prob)):
            # if accept_log_prob < 0: print("accepted new state")
            x, data = x_new, data_new #, fn(x_new, data_new), fn(x, data)
        else:
            pass
    return x, data, fn(x, data), old_log_prob



def train(C, D, N, inits, priors, rand_walk_stds, t_ctrl, tau, n_iter, n_burn_in, m):
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
    params = [1, 1, 1, 1]
    # params[0] = 0.2
    # params[1] = 0.2
    # params[2] = 0.2
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
    assert (E+I > 0).all()
    # P is a list of binomial parameters
    assert (1 >= P).all() and (P >= 0).all()

    # initialize B and params
    print(f"n_burn_in:{n_burn_in}")
    # print("initializing B...")
    # for i in range(n_burn_in):
    #     B, S, E, log_prob_new, log_prob_old = update_data(B, C, D, P, I, S, E, inits, params, N, t_end, t_ctrl, m, epsilon)
    #     assert sum(B) == m
    #     if i % 200 == 0:
    #         params_r = np.round(params, 5)
    #         print(f"iter. {i}=> beta:{params_r[0]}  q:{params_r[1]}  g:{params_r[2]}  gamma:{params_r[3]}  "
    #             )

    # print("initializing params...")
    # for i in range(n_burn_in):
    #     params, P, R0, log_prob_new, log_prob_old = update_params(B, C, D, P, I, S, E, inits, params, priors, rand_walk_stds, N, t_end, t_ctrl, epsilon)
    #     if i % 80 == 0:
    #         params_r = np.round(params, 5)
    #         print(f"iter. {i}=> beta:{params_r[0]}  q:{params_r[1]}  g:{params_r[2]}  gamma:{params_r[3]}  "
    #             )

    # to show final statistics about params
    saved_params = []
    for i in range(n_iter):
        # MCMC update for B, S, E
        B, S, E, log_prob_new, log_prob_old = update_data(B, C, D, P, I, S, E, inits, params, N, t_end, t_ctrl, m, epsilon)
        assert sum(B) == m
        # MCMC update for params and P
        # I is fixed by C and D and doesn't need to be updated
        params, P, R0, log_prob_new, log_prob_old = update_params(B, C, D, P, I, S, E, inits, params, priors, rand_walk_stds, N, t_end, t_ctrl, epsilon)
        if i >= n_burn_in and i % 100 == 0:
            saved_params.append(params)
        if i % 100 == 0:
            params_r = np.round(params+[log_prob_new, log_prob_old, log_prob_new-log_prob_old, params[0]/params[3], sum(R0[-14:-7])], 5)
            print(f"iter. {i}=> beta:{params_r[0]}  q:{params_r[1]}  g:{params_r[2]}  gamma:{params_r[3]}  "
                + f"log prob new:{params_r[4]}  log prob old:{params_r[5]}  diff:{params_r[6]}"
                # + f"log prob diff:{params_r[6]}"
                # + f"  R0:{params_r[7]}")#  R0 -2nd week:{params_r[8]}"
                )
        if i % 50 == 0:
            pass
            # print(f"best B so far:\n{B}")

    return B, np.mean(saved_params, axis=0), np.std(saved_params, axis=0)


def update_data(B, C, D, P, I, S, E, inits, params, N, t_end, t_ctrl, m, epsilon):
    """
    get a sample from p(B|C, D, params) using metropolis hastings
    """

    def fn(x, data):
        """
        x: a sample from p(B|data)
        B(t) has distribution Binom(S(t), P(t))
        returns: log likelihood of p(B|data)
        """
        S, E = data
        # assert (S >= x).all()
        # assert (x >= 0).all()
        # add epsilon to prevent log 0.
        return np.sum(np.log(sp.stats.binom(S, P).pmf(B)+epsilon))
        

    def proposal(x, data, conditions_fn):
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
        S, E = data
        n_tries = 0
        x_new = np.copy(x)
        while n_tries < 1000:
            n_tries += 1
            t_new = np.random.choice(np.nonzero(x_new)[0], min(10, len(np.nonzero(x_new)[0])), replace=False)
            t_tilde = np.random.choice(range(t_end), len(t_new), replace=False)
            
            x_new[t_new] -= 1
            x_new[t_tilde] += 1
            S_new = compute_S(s0, t_end, x_new)
            E_new = compute_E(e0, t_end, x_new, C)

            if conditions_fn(x_new, [S_new, E_new]):
                # assert (S_new >= x_new).all()
                # assert(x_new >= 0).all()
                # print("found new B")
                return x_new, [S_new, E_new]
            else:
                x_new[t_new] += 1
                x_new[t_tilde] -= 1
        # assert (E >= 0).all() and (E+I > 0).all()
        return x, [S, E]

    def conditions_fn(x, data):
        S, E = data
        return sum(x) == m and (E>=0).all() and (E+I>0).all()

    s0, e0, i0 = inits
    data = [S, E]
    B, data, log_prob_new, log_prob_old = metropolis_hastings(B, data, fn, proposal, conditions_fn, burn_in=30)
    return [B] + data + [log_prob_new, log_prob_old]


def update_params(B, C, D, P, I, S, E, inits, params, prior_params, rand_walk_stds, N, t_end, t_ctrl, epsilon):
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
        beta, q, g, gamma = x
        
        # N, t_ctrl, t_end = other_data['N'], other_data['t_ctrl'], other_data['t_end']
        # B, C, D = other_data['B'], other_data['C'], other_data['D']
        # I, S, E = other_data['I'], other_data['S'], other_data['E']

        pC = 1-np.exp(-g)
        pR = 1-np.exp(-gamma)
        P = compute_P(transmission_rate(beta, q, t_ctrl, t_end), I, N)

        # log likelihood
        # add epsilon to prevent log 0.

        logB = np.sum(np.log(sp.stats.binom(S, P).pmf(B)+epsilon))
        logC = np.sum(np.log(sp.stats.binom(E, pC).pmf(C)+epsilon))
        logD = np.sum(np.log(sp.stats.binom(I, pR).pmf(D)+epsilon))

        # assert not np.isnan(logB)
        # assert not np.isnan(logC)
        # assert not np.isnan(logD)

        # log prior
        log_prior = 0
        for i in range(4):
            a, b = prior_params[i]
            log_prior += np.log(sp.stats.gamma(a, b).pdf(x[i])+epsilon)
        # assert not np.isnan(log_prior)
        
        return logB+logC+logD+log_prior

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
        return (x > 0).all()

    # initialize other_data
    # other_data = {}
    # for i in range(len(params)):
    #     other_data['gamma_'+str(i)] = prior_params[i]
    #     other_data['sigma_'+str(i)] = rand_walk_stds[i]

    # params_new = []
    # for i in range(len(params)):
    #     other_data['which_param'] = i
    #     param, _, _, _ = metropolis_hastings(params[i], [params, other_data], fn, proposal, conditions_fn)
    #     params_new.append(param)
    params_new, _, log_prob_new, log_prob_old = metropolis_hastings(np.array(params), None, fn, proposal, conditions_fn)
    
    t_rate = transmission_rate(params_new[0], params_new[1], t_ctrl, t_end)
    return params_new.tolist(), compute_P(t_rate, I, N), t_rate/params_new[3], log_prob_new, log_prob_old

def compute_S(s0, t_end, B):
    """
    S(0) = s0
    S(t+1) = S(t) - B(t) for t >= 0

    can be simplified to S(t+1) = s0 - sum(B[:t])
    """
    return s0 - np.concatenate(([0], np.cumsum(B)[:-1]))


def compute_E(e0, t_end, B, C):
    """
    E(0) = e0
    E(t+1) = E(t) + B(t) - C(t) for t >= 0

    can be simplified to E(t+1) = e0+sum(B[:t]-C[:t])
    """
    return e0 + np.concatenate(([0], np.cumsum(B-C)[:-1]))


def compute_I(i0, t_end, C, D):
    """
    I(0) = i0
    I(t+1) = I(t) + C(t) - D(t) for t >= 0

    can be simplified to I(t+1) = i0+sum(C[:t]-D[:t])
    """
    return i0 + np.concatenate(([0], np.cumsum(C-D)[:-1]))

def transmission_rate(beta, q, t_ctrl, t_end):
    """
    rate of transmission on day t, ie. the number of
    newly infected individuals on day t.
    
    Note: this is different from R0
    """
    trans_rate = np.ones((t_end, )) * beta
    if t_ctrl < t_end:
        ctrl_indices = np.array(range(t_ctrl, t_end))
        trans_rate[ctrl_indices] = beta* np.exp(-q*(ctrl_indices-t_ctrl))

    # try: assert trans_rate.all() >= 0
    # except AssertionError as e:
    #     print(beta, q, trans_rate[trans_rate < 0])
    #     raise e
    return trans_rate

def compute_P(trans_rate, I, N):
    try:
        return 1 - np.exp(-trans_rate * I/N)
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
    t_rate = transmission_rate(beta, q, t_ctrl, tau-1)
    
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
    if sum(B) > 20:
        print(f"number of observations:{t}")
        return sum(B), np.array(C), np.array(D)
    else:
        return create_dataset(inits, beta, q, g, gamma, t_ctrl, tau)



if __name__ == '__main__':
    N = 5364500
    t_end = 100
    inits = [N, 1, 0]
    priors = [(2, 10)]*4
    rand_walk_stds = [0.005, 0.005, 0.005, 0.005]
    t_ctrl = 130
    tau = 1000
    n_iter = 100000
    n_burn_in = 10000
    m, C, D = create_dataset(inits, beta=0.2, q=0.2, g=0.3, gamma=0.1429, t_ctrl=t_ctrl, tau=tau)
    print(train(C, D, N, inits, priors, rand_walk_stds, t_ctrl, tau, n_iter, n_burn_in, m)[1:])