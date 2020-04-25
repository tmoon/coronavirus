from joblib import Parallel, delayed
import requests
import pandas as pd
import itertools
import json

def download_data(arg):
    id1, id2 = arg
    url = "http://www.rhd.gov.bd/OnlineRoadNetwork/getDistanceLocation.asp?OriginID={}&DestinationID={}&Node_1_ID=0&Node_2_ID=0".format(id1, id2)
    print("getting", id1, id2)
    res = requests.get(url)
    arr = [id1, id2, res.text]
    with open("upozilla_dist_dump/upozilla_{}_{}.json".format(id1, id2), 'w') as f:
        json.dump(arr, f)
    
    return arr


if __name__ == '__main__':
    pairs = [x for x in itertools.combinations(list(range(1, 14)), 2)]
    print(len(pairs))
    parallel_worker = Parallel(n_jobs=32, backend='threading', verbose=50)

    res = parallel_worker(delayed(download_data)(arg) for arg in pairs)

    df = pd.DataFrame(res, columns=["id1", "id2", "res"])

    df.to_csv("upozilla_dist_dump/data_v1.csv")