import argparse
import datetime

from communication import GmailCommunication
from database import DatabaseUtils
from inference import Inference
import pandas as pd
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="vicuna-13B-v1")  # You could change this to TheBloke/vicuna-13B-1.1-HF to download a HF checkpoint or even a 7B model
    parser.add_argument("--creds_file", type=str, default="config/creds.json")
    parser.add_argument("--start_date", type=str, default=TODAY.strftime('%Y-%m-%d'))
    parser.add_argument("--end_date", type=str, default=TODAY.strftime('%Y-%m-%d'))
    parser.add_argument("--research", type=str, default="config/research_interests.txt")
    parser.add_argument("--num_gpus", type=int, default=2)  # you may want to change this to 1 if you are with only a single card.
    parser.add_argument("--db_location", type=str, default=None)
    parser.add_argument("--mkdirs", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load_8_bit", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)  # I like loading bars
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("top_p", type=float, default=0.75)
    parser.add_argument("max_generated_tokens", type=int, default=512)

    args = parser.parse_args()
    verbose = args.verbose

    if verbose:
        print("Downloading Paperswithcode Data")
    data = ProcessData(args.start_date, args.end_date)
    data_df = ProcessData.download_and_process_data()
    abstracts = list(data_df["abstract"])
    
    if verbose:
        print("Data downloaded successfully")
    
    if verbose:
        print("Generating connection to sqlite3")
    db_utils = DatabaseUtils(mkdirs=args.mkdirs, db_filename=args.db_location)
    
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
    data_df['summary'] = summaries
    data_df['recommended_text'] = model_ouputs
    data_df['recommended'] = recommendations



