import datetime
import itertools
import math
import matplotlib.dates as mdates
import numpy as np
import operator
import pandas as pd
import pytz
from tqdm import tqdm_notebook as tqm

from IPython.display import display

from collections import OrderedDict
from numpy import sqrt, mean, square, median
from IPython.display import display

import matplotlib.pyplot as plt

# sns and plotting style
import matplotlib as mlp

# mlp.style.use('classic')

import seaborn as sns

# sns.set(style='ticks', palette=sns.color_palette("Set2", 10))
sns.set_style("darkgrid")

mlp.rcParams['axes.titlesize'] = 24
mlp.rcParams['axes.labelsize'] = 20
mlp.rcParams['xtick.labelsize'] = 18
mlp.rcParams['ytick.labelsize'] = 18
mlp.rcParams['figure.figsize'] = 15, 10
mlp.rcParams['legend.fontsize']= 20

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns',500)
pd.set_option('display.width', 3500)

eastern = pytz.timezone('US/Eastern')
# libraries for async results
import time
from multiprocessing import Pool
# import ipython_bell


def ping_bell(msg):
    """
    General osx notification using ipython bell library
    """
    ipython_bell.notifiers.Notification().osx("Finished Job!", "Job: %s" % msg)
    # print msg

_async_res = {}
def _wrapper_fn(*args, **kargs):
    import time
    # TODO: in future, instead of passing msg, this should wrap around the given fn
    # Then modify _log_result to catch that accordingly

    t0 = time.time()

    res = []

    try:
        res = {
            "result": kargs.get("fn")(*args), 
            "time": time.time() - t0, 
            "msg": "Success!", 
            "success": True
            }
    except Exception as e:
        res = {
            "result": None, 
            "time": time.time() - t0, 
            "msg": "Error: %s" % str(e), 
            "success": False
            }
    
    return res

class AsyncRun(object):
    """Run a function in a different thread using parallelism

    Then it notifies you once the job is finished
    
    Dependencies:
    ipython bell: pip install IPythonBell


    Usage:
    - pawn a worker to run a function in parallel
        REQUIREMENT: function input must be of the form <input_object, job_name> and output <output_object, job_name>
    - to add to notebook run the following
        ```
        %matplotlib inline
        
        # -i is important because it must have access to the namespace
        %run -i ../libs/import_libs.py
        ```
    - Once the job is finished you will get a notification with the job_name in the description
    - to access output use `_async_res[job_name]`
    
    - Example:

        def foo_pool(df, msg):
            return df.tid.sum(), msg

        AsyncRun(foo_pool2, df_auto[["tid"]], "sum_tids").run()
    """
    def __init__(self, *args, **kargs):
        super(AsyncRun, self).__init__()
        self.args = args
        self.kargs = kargs
        self.job_name = kargs.get("job_name", "test0")

    def _log_result(self, result):
        r, success = result, result["success"]
        _async_res[self.job_name] = r

        if success:
            ping_bell("%s finished successfully!" % self.job_name)
        else:
            ping_bell("%s FAILED! %s" % (self.job_name, r.get("msg")))

    def run(self):
        pool = Pool(1)
        pool.apply_async(_wrapper_fn, args=self.args, kwds=self.kargs, callback=self._log_result)
        pool.close()
        # pool.join()


def test_fn(arr1, arr2):
    return np.sum(arr1) + np.sum(arr2)
        