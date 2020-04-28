import pandas as pd

last_date = "28 Apr"

def rm_between(s, c1='[', c2=']'):
    if c1 in s and c2 in s:
        return s[:s.find(c1)] + s[s.find(c2) + 1:]
    else:
        return s

def find_between(s, c1='(', c2=')'):
    if c1 in s and c2 in s:
        return s[s.find(c1) + 1: s.find(c2)]
    else:
        return ''

def find_upto_date(df, date_s="28 Apr"):
    idx = list(germany_df.Date).index(date_s) + 1
    
    return df.iloc[:idx].copy().reset_index(drop=1)

    
germany_df = pd.read_html("https://en.wikipedia.org/wiki/Template:2019%E2%80%9320_coronavirus_pandemic_data/Germany_medical_cases")[0]
germany_df = germany_df.replace('—', '')
germany_df.columns = germany_df.columns.get_level_values(1)

germany_df = find_upto_date(germany_df, date_s=last_date)

germany_df.to_csv("germany_uncleaned_data_%s.csv" % last_date, index=0)

german_case_df = germany_df.copy()
for c in german_case_df.columns[1:]:
    german_case_df[c] = [rm_between(str(s)).split("(")[0] for s in german_case_df[c]]

german_case_df.to_csv("germany_case_data_%s.csv" % last_date, index=0)

germany_df
german_death_df = germany_df.copy()
for c in german_death_df.columns[1:]:
    german_death_df[c] = [find_between(str(s)).split("(")[0] for s in german_death_df[c]]

german_death_df.to_csv("germany_death_data_%s.csv" % last_date, index=0)