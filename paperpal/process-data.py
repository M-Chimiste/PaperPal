import requests
import json
import gzip
import os
import datetime
import shutil
import pandas as pd


# Surpisingly it's faster and easier to download dump than query an API for the most current data.
# Go figure... Please fix this paperswithcode...
os.makedirs('temp_data', exist_ok=True)
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


