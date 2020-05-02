import argparse
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='Live Rt Estimation')
parser.add_argument(
    '--rtmax',
    type=int,
    default=12,
    help='max allowed value of Rt (default: %(default)d)',
    nargs='?',
)
parser.add_argument(
    '--gamma', type=float, help='1/serial interval (default: %(default)f)', nargs='?', default=1 / 7
)
parser.add_argument(
    '--infile',
    type=str,
    help='input file name. the format should be <country>_<last date of observation>.csv (default: %(default)s)',
    nargs='?',
    default='germany_april_27.csv',
)
parser.add_argument(
    '--indir',
    type=str,
    help='input file directory (default: %(default)s)',
    nargs='?',
    default='datasets',
)
parser.add_argument(
    '--outdir',
    type=str,
    help='output file directory (default: %(default)s)',
    nargs='?',
    default='rt',
)
parser.add_argument(
    '--cutoff',
    type=int,
    help='threshold for number of new positive cases to be detected on a single day. estimation begins from that day. (default: %(default)d)',
    nargs='?',
    default=25,
)

args = parser.parse_args()

# We create an array for every possible value of Rt
R_T_MAX = args.rtmax
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
GAMMA = args.gamma
filedir = args.indir
filename = args.infile
country_name = '_'.join(filename.split('_')[:-2])
cutoff = args.cutoff
outdir = args.outdir

print(f"estimating Rt for {country_name}...")


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


country = pd.read_csv(
    filedir + '/' + filename,
    usecols=[0, 1],
    parse_dates=[0],
    index_col=[0],
    names=['date', 'positive'],
    header=None,
    skiprows=1,
    squeeze=False,
).sort_index()  #


def prepare_cases(cases, cutoff=25):
    new_cases = cases.diff()
    smoothed = (
        new_cases.rolling(7, win_type='gaussian', min_periods=1, center=True).mean(std=2).round()
    )
    idx_start = np.searchsorted(smoothed, cutoff)

    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    return original, smoothed


cases = country['positive'].rename(f"{country_name} cases")
original, smoothed = prepare_cases(cases, cutoff)

original.plot(
    title=f"{country_name} New Cases per Day",
    c='k',
    linestyle=':',
    alpha=0.5,
    label='Actual',
    legend=True,
    figsize=(500 / 72, 300 / 72),
)

ax = smoothed.plot(label='Smoothed', legend=True)

ax.get_figure().set_facecolor('w')
ax.get_figure().savefig(f"new_cases_{country_name}.png", dpi=200)


def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(sr[1:].values, lam), index=r_t_range, columns=sr.index[1:]
    )

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range, scale=sigma).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    # prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range) / len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(index=r_t_range, columns=sr.index, data={sr.index[0]: prior0})

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


# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = get_posteriors(smoothed, sigma=0.25)

# Note that this takes a while to execute - it's not the most efficient algorithm
hdis = highest_density_interval(posteriors, p=0.9)
most_likely = posteriors.idxmax().rename('ML')
# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

sigmas = np.linspace(1 / 20, 1, 20)

new, smoothed = prepare_cases(cases, cutoff=cutoff)
result = {}

# Holds all posteriors with every given value of sigma
result['posteriors'] = []

# Holds the log likelihood across all k for each value of sigma
result['log_likelihoods'] = []

for sigma in sigmas:
    posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
    result['posteriors'].append(posteriors)
    result['log_likelihoods'].append(log_likelihood)


total_log_likelihoods = result['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index = np.argmax(total_log_likelihoods)

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]

final_results = None

posteriors = result['posteriors'][max_likelihood_index]
hdis_90 = highest_density_interval(posteriors, p=0.9)
hdis_50 = highest_density_interval(posteriors, p=0.5)
most_likely = posteriors.idxmax().rename('ML')
result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)

if final_results is None:
    final_results = result
else:
    final_results = pd.concat([final_results, result])

result.iloc[1:].to_csv(f'{outdir}/rt_{country_name}.csv')
print(f"output saved to {outdir}/rt_{country_name}.csv")
