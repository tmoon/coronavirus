import numpy as np 
import scipy as sp
import pandas as pd
from scipy import stats, optimize, interpolate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

np.seterr(all='ignore')
warnings.filterwarnings('ignore')

import time

"""
The model learns its parameters from C and D. see docstring of train()
These parameters can be used for R0 estimation and for making other 
predictions.

I generated a dummy dataset that was in paper section 3.3. I set 
N=s0=500 instead of 5364500 for speed. see __name__ == __main__:
"""

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
    return e0 + np.concatenate(([0], np.cumsum(B - C)[:-1]))


def metropolis_hastings(x, data, fn, proposal, conditions_fn, burn_in=1, interval=1, num_samples=1):
    """
    get num_samples samples from a distribution p(x) ~ k*fn(x) given proposal
    distribution proposal(x) with metropolis hastings algorithm

        * the new sample has to satisfy the conditions in conditions_fn
        * data is a list of additional distribution, variables etc that are
          required to compute the functions
        * assumes proposal distribution is symmetric, ie: q(x'|x) = q(x|x')
        * fn returns log prob. for numeric stability

    returns: num_samples samples from p(x) and corresponding data
    """
    sampled_x = 0
    for i in range(burn_in+interval*num_samples+1):
        x_new, data_new = proposal(x, data, conditions_fn)
        accept_log_prob = min(0, fn(x_new, data_new) - fn(x, data))
        if np.random.binomial(1, np.exp(accept_log_prob)):
            x, data = x_new, data_new
        # else reject the sample
        
        # now save one sample out of interval samples
        if i > burn_in and (i-burn_in) % interval == 0:
            sampled_x += x
    return sampled_x/num_samples


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
            t_new = np.random.choice(np.nonzero(x_new)[0], min(30, len(np.nonzero(x_new)[0])), replace=False)
            t_tilde = np.random.choice(range(t_end), len(t_new), replace=False)
            
            x_new[t_new] -= 1
            x_new[t_tilde] += 1
            S_new = compute_S(s0, t_end, x_new)
            E_new = compute_E(e0, t_end, x_new, C)

            if conditions_fn(x_new, [S_new, E_new]):
                # assert (S_new >= x_new).all()
                # assert(x_new >= 0).all()
                return x_new, [S_new, E_new]
            else:
                # revert back the changes
                x_new[t_new] += 1
                x_new[t_tilde] -= 1
        # assert (E >= 0).all() and (E+I > 0).all()
        return x, [S, E]

    def conditions_fn(x, data):
        S, E = data
        return np.sum(x) == m and (E>=0).all() and (E+I>0).all()

    s0, e0, i0 = inits
    data = [S, E]
    
    log_prob_old = fn(B, data)
    B = metropolis_hastings(B, data, fn, proposal, conditions_fn, burn_in=50000, interval=5, num_samples=40)
    B = np.floor(B+0.5).astype(int)
    residue = m-np.sum(B)
    assert np.abs(residue) <= len(B)
    B[:np.abs(residue)] += np.sign(residue)
    data = compute_S(s0, t_end, B), compute_E(e0, t_end, B, C)
    log_prob_new = fn(B, data)
    # print(m, np.sum(B))
    return [B] + [data[0], data[1]] + [log_prob_new, log_prob_old]