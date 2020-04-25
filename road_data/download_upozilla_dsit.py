from joblib import Parallel, delayed
import requests
import pandas as pd
import itertools
import json

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def requests_retry_session(
    retries=5,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session


def download_data(arg):
    id1, id2 = arg
    url = "http://www.rhd.gov.bd/OnlineRoadNetwork/getDistanceLocation.asp?OriginID={}&DestinationID={}&Node_1_ID=0&Node_2_ID=0".format(id1, id2)
    # print("getting", id1, id2)
    arr = [id1, id2, 'FAILED']
    try:
        response = requests_retry_session().get(url, timeout=10)
    except Exception as x:
        print('It failed :', x.__class__.__name__)
    finally:
        if response.status_code == 200:
            arr = [id1, id2, response.text]
            response.connection.close()

            with open("upozilla_dist_dump/upozilla_{}_{}.json".format(id1, id2), 'w') as f:
                json.dump(arr, f)
    
    return arr


if __name__ == '__main__':
    pairs = [x for x in itertools.combinations(list(range(1, 100)), 2)]
    print(len(pairs))
    parallel_worker = Parallel(n_jobs=32, backend='threading', verbose=50)

    res = parallel_worker(delayed(download_data)(arg) for arg in pairs)

    df = pd.DataFrame(res, columns=["id1", "id2", "res"])

    df.to_csv("upozilla_dist_dump/data_v1.csv")