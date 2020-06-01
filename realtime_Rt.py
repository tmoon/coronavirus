import numpy as np
import pandas as pd

from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates

from scipy import stats as sps
from scipy.interpolate import interp1d


def batch_estimate_rt(data_list, region_name_list, serial_interval=7, cutoff=10, rtmax=12, ci_pct=0.9):
    '''
    this is just a wrapper around estimate_rt. see the docstring for estimate_rt for documentation
    
    data_list:        a list of data parameter accepted by estimate_rt
    region_name_list: a list of region_name parameter accepted by estimate_rt

    return: a list of rts for each region in data_list
    '''
    assert(len(data_list) == len(region_name_list))
    rt_list = []
    for data, region_name in zip(data_list, region_name_list):
        rt = estimate_rt(data, region_name, serial_interval, cutoff, rtmax, ci_pct)
        rt_list.append(rt)

    return rt_list



def estimate_rt(data, region_name, serial_interval=7, cutoff=10, rtmax=12, ci_pct=0.9):
    '''
    data: pandas DataFrame of date and number of tested positives (cumulative). read data from csv with 
          the following command

    data = pd.read_csv(
        filepath
        usecols=[0, 1],
        parse_dates=[0],
        index_col=[0],
        names=['date', 'positive'],
        header=None,
        skiprows=1,
        squeeze=False,
    ).sort_index() 

    serial_interval: serial interval of covid -19
    cutoff:          threshold for number of new positive cases to be detected on a single day.
    rtmax:           max allowed value for Rt   
    
    returns: a pandas Dataframe of date, ML estimate of Rt and ci_pct error bounds. (default: 90%) 
    '''
    gamma = 1/serial_interval
    rt_range = np.linspace(0, rtmax, rtmax * 100 + 1)

    
    print(f"estimating Rt for {region_name}...")

    cases = data['positive']
    sigmas = np.linspace(1 / 20, 1, 20)

    new, smoothed = prepare_cases(cases, cutoff=cutoff)
    result = {}

    # Holds all posteriors with every given value of sigma
    result['posteriors'] = []

    # Holds the log likelihood across all k for each value of sigma
    result['log_likelihoods'] = []

    for sigma in sigmas:
        posteriors, log_likelihood = get_posteriors(smoothed, gamma, rt_range, sigma=sigma)
        result['posteriors'].append(posteriors)
        result['log_likelihoods'].append(log_likelihood)


    total_log_likelihoods = result['log_likelihoods']

    # Select the index with the largest log likelihood total
    max_likelihood_index = np.argmax(total_log_likelihoods)

    # Select the value that has the highest log likelihood
    sigma = sigmas[max_likelihood_index]

    posteriors = result['posteriors'][max_likelihood_index]
    hdis = highest_density_interval(posteriors, p=ci_pct)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis], axis=1)

    return result.iloc[1:].reset_index()


def highest_density_interval(pmf, p=0.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if isinstance(pmf, pd.DataFrame):
        return pd.DataFrame(
            [highest_density_interval(pmf[col], p=p) for col in pmf], index=pmf.columns
        )

    cumsum = np.cumsum(pmf.values)
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()

    # Find the smallest range (highest density)
    best = (highs - lows).argmin()

    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]

    return pd.Series([low, high], index=[f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])


def prepare_cases(cases, cutoff=25):
    new_cases = cases.diff()
    smoothed = (
        new_cases.rolling(7, win_type='gaussian', min_periods=1, center=True).mean(std=2).round()
    )
    idx_start = np.searchsorted(smoothed, cutoff)

    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    return original, smoothed

def get_posteriors(sr, gamma, rt_range, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(gamma * (rt_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(sr[1:].values, lam), index=rt_range, columns=sr.index[1:]
    )

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=rt_range, scale=sigma).pdf(rt_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    # prior0 = sps.gamma(a=4).pdf(rt_range)
    prior0 = np.ones_like(rt_range) / len(rt_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(index=rt_range, columns=sr.index, data={sr.index[0]: prior0})

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        # (5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        # (5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood


data = pd.read_csv(
        "datasets/bd_april_22.csv",
        usecols=[0, 1],
        parse_dates=[0],
        index_col=[0],
        names=['date', 'positive'],
        header=None,
        skiprows=1,
        squeeze=False,
    ).sort_index()

print(estimate_rt(data, region_name="bd"))