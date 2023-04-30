import argparse
import datetime
import os
import sqlite3

import pandas as pd
from communication import GmailCommunication
from inference import Inference
from processdata import ProcessData
from tqdm import tqdm

TODAY = datetime.date.today()


def get_research_interests(filename):
    """Function to generate the research interests information.

    Args:
        filename (str): Filename of your text file with research interests.

    Returns:
        str: concatenated research interests by newline.
    """
    with open(filename, 'r') as f:
        interests = f.readlines()
    interests = '/n'.join(interests)
    return interests


def create_connection(db_filename):
        """Function to initialize the connection to the db.  Note that this will create a file
        if one does not already exist

        Args:
            db_filename (str): Path of the database file.

        Returns:
            sqlite connection: connection object for the sqlite database.
        """
        try:
            conn = sqlite3.connect(db_filename)
            return conn
        except Exception as e:
            print(e)


def attempt_to_get_recommendation(text):
    """Function to quickly get a recommendation.  I tried to prompt the LLM to give me yes/no.
    TODO - Potentially replace this as a classifier
    Args:
        text (str): Model output from LLM

    Returns:
        str: a yes or no
    """
    text = text.lower()
    if 'yes' in text:
        recommendation = 'yes'
    if 'no' in text:
        recommendation = 'no'
    else:
        recommendation = 'unk' # I want to manually review these in my daily df.

    return recommendation


def get_desired_content(data_df, recommended):
    """Function to format the data from the filtered dataframe.

    Args:
        data_df (df): Data of interest to extract
        recommended (str): if the text was recommended or not or unk.

    Returns:
        str: content of interest
    """
    summaries = list(data_df["summary"])
    titles = list(data_df["title"])
    recommendation_text = list(data_df["recommended_text"])
    urls = list(data_df["url_pdf"])
    content = [f"The following papers have a recommendation of: {recommended}"]
    for title, summary, recommendation, url in zip(titles, summaries, recommendation_text, urls):
        line = f"{title}: {summary} | {recommendation} | {url}"
        content += line
    content = '/n'.join(content)
    return content


def construct_email_body(recommended, unk, not_recommended):
    body = f"""Hello there!  Here is your paper content you requested!
    These are the following papers I think you might want to look at:
    {recommended}

    These are the papers I wasn't as sure about:
    {unk}

    These are the papers I didn't think you would want to look at:
    {not_recommended}

    Have a wonderful day,
    ~PaperPal~
    """
    return body


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="models/vicuna-13B-v1")  # You could change this to TheBloke/vicuna-13B-1.1-HF to download a HF checkpoint or even a 7B model
    parser.add_argument("--creds_file", type=str, default="config/creds.json")
    parser.add_argument("--start_date", type=str, default=TODAY.strftime('%Y-%m-%d'))
    parser.add_argument("--end_date", type=str, default=TODAY.strftime('%Y-%m-%d'))
    parser.add_argument("--research", type=str, default="config/research_interests.txt")
    parser.add_argument("--num_gpus", type=int, default=2)  # you may want to change this to 1 if you are with only a single card.
    parser.add_argument("--db_location", type=str, default="database/paperpal_sqlite.db")  # I will handle making this directory for you.  If you want to put this somewhere else you need to handle making the folder
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load_8_bit", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)  # I like loading bars
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--max_generated_tokens", type=int, default=512)
    parser.add_argument("--sender_address", default=None)
    parser.add_argument("--receiver_address", default=None)

    args = parser.parse_args()
    verbose = args.verbose

    if args.db_location is "database/paperpal_sqlite.db":
        os.makedirs('database', exist_ok=True)

    if verbose:
        print("Downloading Paperswithcode Data")
    data = ProcessData(args.start_date, args.end_date)
    data_df = ProcessData.download_and_process_data()
    abstracts = list(data_df["abstract"])
    
    if verbose:
        print("Data downloaded successfully")
    
    if verbose:
        print("Generating connection to sqlite3")
    
    db_connection = create_connection(args.db_location)
    
    if verbose:
        print("Beginning model load")

    llm = Inference(model_name=args.model,
                    device=args.device,
                    num_gpus=args.num_gpus,
                    load_8bit=args.load_8_bit)
    
    summaries = []
    model_ouputs = []
    recommendations = []

    research_interests = get_research_interests(args.research)

    for abstract in tqdm(abstracts, disable=not verbose):
        summary_prompt = llm.construct_summary_prompt
        summary = llm.generate(text=summary_prompt,
                               temp=args.temp,
                               top_k=args.top_k,
                               top_p=args.top_p,
                               num_beams=args.num_beams,
                               max_tokens=args.max_generated_tokens)
        summaries.append(summary)
        model_ouput_prompt = llm.construct_research_prompt(summary, research_interests)
        model_output = llm.generate(text=model_ouput_prompt,
                               temp=args.temp,
                               top_k=args.top_k,
                               top_p=args.top_p,
                               num_beams=args.num_beams,
                               max_tokens=args.max_generated_tokens)
        model_ouputs.append(model_output)
        recommendation = attempt_to_get_recommendation(recommendation)
        recommendations.append(recommendation)

    if verbose:
        print("Creating merged df")

    data_df["summary"] = summaries
    data_df["recommended_text"] = model_ouputs
    data_df["recommended"] = recommendations

    data_df.to_sql('papers', con=db_connection, if_exists='append')  # send data to sqlite3

    recommended = data_df.loc[data_df['recommended'] == "yes"]
    not_recommended = data_df.loc[data_df['recommended'] == "no"]
    unk_data = data_df.loc[data_df['recommended'] == "unk"]

    recommended_text = get_desired_content(recommended, 'yes')
    not_recommended_text = get_desired_content(not_recommended, 'no')
    unk_data_text = get_desired_content(unk_data, 'unk')

    body = construct_email_body(recommended=recommended_text,
                                unk=unk_data_text, 
                                not_recommended=not_recommended_text)

    if verbose:
        print("Sending an email")
    
    gmail = GmailCommunication(sender_address=args.sender_address, receiver_address=args.receiver_address, creds_path=args.creds_file)
    gmail.compose_message(content=body)
    gmail.send_email()

    if verbose:
        print("Cleaning up...")
    
    data.cleanup_temp_and_mem()

    





