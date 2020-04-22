from joblib import Parallel, delayed
import pandas as pd

import os
from PyPDF2 import PdfFileReader


def find_c2_pages(fileReader):
    flag = False
    i = 1
    
    arr_dict = {}
    page_arr = []
    s = "Table C-%.2d" % i

    for pn in range(65, fileReader.getNumPages()):

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

    return arr_dict

def read_meta(dist_name):
    fname = 'census_pdfs/%s.pdf' % dist_name
    file = open(fname, 'rb')

    # creating a pdf reader object
    fileReader = PdfFileReader(file)

    arr_dict = find_c2_pages(fileReader)

print(read_meta("BAGERHAT"))
# parallel_worker = Parallel(n_jobs=16, backend='multiprocessing', 
#     verbose=10)
# parallel_worker(delayed(self.sql_fn)(arg) for arg in self.arg_array)
#         df = pd.concat(df_arr)