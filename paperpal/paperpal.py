import argparse
import datetime
import json
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


def construct_email_body(recommended,
                         unk,
                         not_recommended,
                         start_date,
                         end_date):
    if start_date == end_date:
        date_range = start_date
    else:
        date_range = f"{start_date} - {end_date}"
    body = f"""Hello there!  Here is your paper content you requested for {date_range}!
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
    parser.add_argument("--config", default=None)

    args = parser.parse_args()
    args = vars(args)

    if args.get("config"):
        print("Configuration file used defaults will be overwritten by config file.")

    if args.get("config"):
        try:
            with open(args.get("config"), 'r') as f:
                config_json = json.load(f)
            # merge the two dictionaries
            args.update(config_json)
        except Exception as e:
            print("Error loading creds file: ", str(e))
            raise e


    verbose = args.get("verbose")

    if verbose:
        print("Downloading Papers with Code Data")
    data = ProcessData(args.get("start_date"), args.get("end_date"))
    data_df = data.download_and_process_data()
    abstracts = list(data_df["abstract"])
    
    if verbose:
        print(f"Data downloaded successfully. Found {len(abstracts)} papers.")
    
    
    if verbose:
        print("Beginning model load")

    llm = Inference(model_name=args.get("model"),
                    device=args.get("device"),
                    num_gpus=args.get("num_gpus"),
                    load_8bit=args.get("load_8_bit"))
    
    summaries = []
    model_ouputs = []
    recommendations = []
    rating = []

    research_interests = get_research_interests(args.get("research"))

    for abstract in tqdm(abstracts, disable=not verbose):
        
        model_output_prompt = llm.construct_research_prompt(abstract, research_interests)
        model_output = llm.generate(text=model_output_prompt,
                               temp=args.get("temp"),
                               top_k=args.get("top_k"),
                               top_p=args.get("top_p"),
                               num_beams=args.get("num_beams"),
                               max_tokens=args.get("max_generated_tokens"))
        
        

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
    
    if args.get("csv"):
        os.makedirs("csv_output", exist_ok=True)
        if args.get("start_date") == args.get("end_date"):
            date_range = args.get("start_date")
        else:
            start_date = args.get("start_date")
            end_date = args.get("end_date")
            date_range = f"{start_date}-{end_date}"
        csv_output = f"csv_output/{date_range}-paperpal-output.csv"
        data_df.to_csv(csv_output)

    recommended = data_df.loc[data_df['recommended'] == "yes"]
    not_recommended = data_df.loc[data_df['recommended'] == "no"]
    unk_data = data_df.loc[data_df['recommended'] == "unk"]

    recommended_text = get_desired_content(recommended, 'yes')
    not_recommended_text = get_desired_content(not_recommended, 'no')
    unk_data_text = get_desired_content(unk_data, 'unk')

    body = construct_email_body(recommended=recommended_text,
                                unk=unk_data_text, 
                                not_recommended=not_recommended_text,
                                start_date=args.get("start_date"),
                                end_date=args.get("end_date"))

    if verbose:
        print("Sending an email")
    
    gmail = GmailCommunication(sender_address=args.get("sender_address"), receiver_address=args.get("receiver_address"), creds_path=args.get("creds_file"))
    gmail.compose_message(content=body, start_date=args.get("start_date"), end_date=args.get("end_date"))
    gmail.send_email()

    if verbose:
        print("Cleaning up...")
    
    data.cleanup_temp_and_mem()

    





