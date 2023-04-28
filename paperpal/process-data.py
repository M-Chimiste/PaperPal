import requests
import json
import gzip
import os
import datetime
import shutil
import pandas as pd


# Surpisingly it's faster and easier to download dump than query an API for the most current data.
# Go figure... Please fix this paperswithcode...
os.makedirs('temp_data', exist_ok=True)  # Generate a temp_data repository
PAPERS_DUMP = "https://production-media.paperswithcode.com/about/papers-with-abstracts.json.gz"  #This URL is static and updated daily per https://github.com/paperswithcode/paperswithcode-data


def fetch_data():
    """Function will download and extract the json data from papers with code

    Returns:
        list: List of dictionaries of the loaded data.
    """
    today = datetime.date.today()
    str_time = today.strftime("%Y-%m-%d")
    gzip_filename = f"temp_data/papers-with-code-{str_time}.json.gz"
    filename = f"temp_data/papers-with-code-{str_time}.json"
    
    with open(gzip_filename, "wb") as fb:
        response = requests.get(PAPERS_DUMP)
        fb.write(response.content)

    with gzip.open(gzip_filename, "rb") as fr:
        with open(filename, "wb") as f_out:
            shutil.copyfileobj(fr, f_out)
    
    with open(filename, 'r') as f:
        json_data = json.load(f)
    
    return json_data


def find_specific_date_data(start_date, end_date, json_dict):
    """function to get specific data from the json data by a date range.
    Dates can be either a range or the same.

    Args:
        start_date (datetime): The desired start date.
        end_date (datetime): The desired end date.  Can be the same as start date.
        json_dict (list): List of dictionaries from the loaded papers with code data

    Returns:
        df: Pandas Dataframe of the associated data.
    """
    
    if start_date == end_date:
        one_day = True
    else:
        one_day = False
    
    df = pd.DataFrame.from_dict(json_dict)

    if one_day:
        df = df.loc[df['published'] == start_date ]
    else:
        df = df.loc[(df['published'] >= start_date) & (df['published'] <= end_date)]

    return df


def cleanup_temp():
    shutil.rmtree('temp_data')