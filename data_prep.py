import pandas as pd
from siuba.dply.forcats import fct_lump


train_x = pd.read_csv("./data/raw/train_x.csv", parse_dates=['date_recorded'])
train_y = pd.read_csv("./data/raw/train_y.csv")
test_x = pd.read_csv("./data/raw/test_x.csv", parse_dates=['date_recorded'])

dict_df = {"train_x": train_x,
           "test_x": test_x}
           
for key in dict_df:

    df = dict_df[key]

    # Initialize list of columns to keep
    cols_to_keep = df.columns.tolist()

    to_drop = ['date_recorded', 'recorded_by']
    cols_to_keep = [c for c in cols_to_keep if c not in to_drop]

    # Geo cols to remove
    cols_to_keep.remove('region_code')
    cols_to_keep.remove('district_code')
    cols_to_keep.remove('wpt_name')
    cols_to_keep.remove('subvillage')
    cols_to_keep.remove('ward')

    # Aggregated cols to remove
    aggs_to_drop = ['extraction_type', 'extraction_type_group', 'management', 'payment', 'water_quality',
                   'quantity', 'source', 'source_type', 'waterpoint_type', 'scheme_name']

    cols_to_keep = [c for c in cols_to_keep if c not in aggs_to_drop]

    df = df.loc[:, cols_to_keep]


    # Change booleans to strings
    df["permit"] = df["permit"].astype("string")
    df["public_meeting"] = df["public_meeting"].astype("string")

    # Missing data - code as "missing" for categorical data
    df['funder'] = df['funder'].fillna("missing")
    df['installer'] = df['installer'].fillna("missing")
    df['public_meeting'] = df['public_meeting'].fillna("missing")
    df['scheme_management'] = df['scheme_management'].fillna("missing")
    df['permit'] = df['permit'].fillna("missing")


    # Factor level lumping

    df['funder'] = fct_lump(df['funder'], n=10)
    df['installer'] = fct_lump(df['installer'], n=10)
    df['scheme_management'] = fct_lump(df['scheme_management'], n=10)
    
    df.to_json(f"data/{key}.json", orient='records')
    df.to_csv(f"data/{key}.csv", index=False)
