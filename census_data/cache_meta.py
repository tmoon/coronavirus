from joblib import Parallel, delayed
import pandas as pd
import json
import os
from PyPDF2 import PdfFileReader


def find_c2_pages(fileReader):
    flag = False
    i = 1
    
    arr_dict = {}
    page_arr = []
    s = "Table C-%.2d" % i

    for pn in range(30, fileReader.getNumPages()):

        print(pn, flag)
        page = fileReader.getPage(pn)
        page_content = page.extractText()
        text = page_content.encode('utf-8')

        
        if s in str(text):
            flag = True
            print("FOUND:", s, pn,)
            page_arr.append(pn)
        elif flag:
            if i > 15:
                break
            
            arr_dict[i] = page_arr
            flag = False
            i += 1
            s = "Table C-%.2d" % i
            page_arr = []

            if s in str(text):
                flag = True
                print("FOUND:", s, pn,)
                page_arr.append(pn)


    return arr_dict

def read_meta(dist_name):
    fname = '/Users/tarik/Dropbox/edo_dsci/misc/census_data/%s.pdf' % dist_name
    file = open(fname, 'rb')

    # creating a pdf reader object
    fileReader = PdfFileReader(file)

    arr_dict = find_c2_pages(fileReader)

    with open("./census_data_meta/%s.json" % dist_name, 'w') as f:
        json.dump(arr_dict, f)

    return arr_dict


if __name__ == '__main__':
    read_meta("NATORE")
    # dist_df = pd.read_csv("./census_data_meta/census_data.csv")

    # # for _, r in link_df2.iterrows():
    # #     os.rename("census_data/%s.pdf" % r.dist_name, "census_data/%s.pdf" % r.clean_dist_name) 

    # parallel_worker = Parallel(n_jobs=8, backend='multiprocessing', 
    #     verbose=50)

    # parallel_worker(delayed(read_meta)(arg) for arg in dist_df.clean_dist_name)