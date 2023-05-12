import argparse
import datetime
import os

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
    interests = '\n'.join(interests)
    return interests


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
    urls = list(data_df["url_pdf"])
    content = []
    for title, summary, url in zip(titles, summaries, urls, ):
        line = f"\n --------------\n{title} | {url}\n{summary}\n --------------\n"
        content += line
    content = ''.join(content)
    return content


def construct_email_body(recommended, unk, not_recommended):
    body = f"""Hello there!  Here is your paper content you requested!
These are the following papers I think you might want to look at:
{recommended}

These are the papers I didn't think you would want to look at:
{not_recommended}

These are the papers I wasn't as sure about:
{unk}

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
    parser.add_argument("--receiver_address", nargs="*", default=[])
    parser.add_argument("--csv", type=bool, default=True)

    args = parser.parse_args()
    verbose = args.verbose

    if verbose:
        print("Downloading Papers with Code Data")
    data = ProcessData(args.start_date, args.end_date)
    data_df = data.download_and_process_data()
    abstracts = list(data_df["abstract"])
    
    if verbose:
        print(f"Data downloaded successfully. Found {len(abstracts)} papers.")
    
    
    if verbose:
        print("Beginning model load")

    llm = Inference(model_name=args.model,
                    device=args.device,
                    num_gpus=args.num_gpus,
                    load_8bit=args.load_8_bit)
    
    summaries = []
    model_ouputs = []
    recommendations = []
    rating = []

    research_interests = get_research_interests(args.research)

    for abstract in tqdm(abstracts, disable=not verbose):
        
        model_output_prompt = llm.construct_research_prompt(abstract, research_interests)
        model_output = llm.generate(text=model_output_prompt,
                               temp=args.temp,
                               top_k=args.top_k,
                               top_p=args.top_p,
                               num_beams=args.num_beams,
                               max_tokens=args.max_generated_tokens)
        
        

        if '"related": true' in model_output:
            model_output = model_output.replace('"related": true', '"related": True')
        if '"related": false' in model_output:
            model_output = model_output.replace('"related": false', '"related": False')
        try:
            output_dict = eval(model_output)  #should be a dict of related: bool, reasoning ; str
            recommendation = output_dict['related']
            
            # Sometimes Vicuna get's case of true/false wrong so let's make everything a dictionary
            if recommendation == True:
                recommendation = 'yes'
            else:
                recommendation = 'no'

            summary = output_dict['reasoning']
        except Exception:
            recommendation = 'UNK'
            summary = "UNK"
            rate = 0
        model_ouputs.append(model_output)
        recommendations.append(recommendation)
        summaries.append(summary)

    if verbose:
        print("Creating merged df")

    data_df["recommended"] = recommendations
    data_df["summary"] = summaries
    data_df["model_output"] = model_ouputs
    
    if args.csv:
        os.makedirs("csv_output", exist_ok=True)
        today = datetime.date.today()
        today_string = today.strftime('%Y-%m-%d')
        csv_output = f"csv_output/{today_string}-paperpal-output.csv"
        data_df.to_csv(csv_output)

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

    





