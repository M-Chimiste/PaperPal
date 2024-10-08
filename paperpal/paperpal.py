import os
import json
import datetime

import pandas as pd
import json_repair
from .communication import GmailCommunication, construct_email_body
from .paperswithcode import ProcessData
from .data_handling import PaperDatabase, Paper, Newsletter
from tqdm import tqdm
from dotenv import load_dotenv
from .prompts import (
    NEWSLETTER_SYSTEM_PROMPT,
    RESEARCH_INTERESTS_SYSTEM_PROMPT,
    newsletter_prompt,
    research_prompt, #TODO You need to test each of these to figure out what you like better
    research_interests_prompt
)
from .inference import (
    SentenceTransformerInference,
    LocalCPPInference,
    LocalCudaInference,
    OpenAIInference,
    AnthropicInference
) 
from .utils import cosine_similarity, get_n_days_ago, TODAY
from .data_handling import PaperDatabase

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
LLAMA_CPP_TOKENIZER = os.getenv("LLAMA_CPP_TOKENIZER", None)


class PaperPal:
    #TODO: Run evaluation on papers
    #TODO: apply top_n cutoff
    #TODO: Generate newsletter
    #TODO: Send newsletter email
    #TODO: Save outpputs to db
    def __init__(self,
                 research_interests_path="config/research_interests.txt",
                 n_days=7,
                 top_n=10,
                 model_type="local",
                 model_name="NousResearch/Hermes-3-Llama-3.1-8B",
                 embedding_model_name="Alibaba-NLP/gte-base-en-v1.5",
                 trust_remote_code=True,
                 receiver_address=None,
                 max_new_tokens=1024,
                 temperature=0.1,
                 cosine_similarity_threshold=0.5,
                 data_path="data/papers.db",
                 verbose=True):
        self.verbose = verbose
        self.research_interests_path = research_interests_path
        self.start_date = get_n_days_ago(n_days)
        self.end_date = TODAY
        self.top_n = top_n
        self.model_type = model_type
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.communication = GmailCommunication(sender_address=os.getenv('GMAIL_SENDER_ADDRESS', None),
                                               app_password=os.getenv('GMAIL_APP_PASSWORD', None),
                                               receiver_address=receiver_address)
        self.papers_db = PaperDatabase(data_path)
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformerInference(embedding_model_name, trust_remote_code=trust_remote_code)
        self.cosine_similarity_threshold = cosine_similarity_threshold
        # Load research interests
        try:
            with open(self.research_interests_path, 'r') as file:
                self.research_interests = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"The research interests file at {self.research_interests_path} could not be found. Please check the path and try again.")
        except IOError:
            raise IOError(f"There was an error reading the file at {self.research_interests_path}. Please check the file permissions and try again.")

        # Load model
        model_type = os.getenv("MODEL_TYPE", "local")
        if model_type == "local":
            from .inference import LocalCudaInference
            import torch
            if torch.cuda.is_available():
                self.inference = LocalCudaInference(model_name, max_new_tokens, temperature)
            else:
                self.inference = LocalCPPInference(model_name, max_new_tokens, temperature, LLAMA_CPP_TOKENIZER)
        elif model_type == "anthropic":
            if ANTHROPIC_API_KEY is None:
                raise ValueError("Anthropic API key is not set. Please check your .env file and ensure ANTHROPIC_API_KEY is properly configured.")
            from .inference import AnthropicInference
            self.inference = AnthropicInference(model_name, max_new_tokens, temperature)
        elif model_type == "openai":
            if OPENAI_API_KEY is None:
                raise ValueError("OpenAI API key is not set. Please check your .env file and ensure OPENAI_API_KEY is properly configured.")
            from .inference import OpenAIInference
            self.inference = OpenAIInference(model_name, max_new_tokens, temperature)
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of 'local', 'anthropic', or 'openai'.")
    

    def download_and_process_papers(self):
        """
        Downloads papers from PapersWithCode based on research interests and date range.
        """
        process_data = ProcessData(start_date=self.start_date, end_date=self.end_date)
        
        data_df = process_data.download_and_process_data(start_date=self.start_date, end_date=self.end_date)

        abstracts = list(data_df['abstract'])
        abstract_embeddings = []
        cosine_similarities = []
        reserch_embedding = self.embedding_model.invoke(self.research_interests)
        for abstract in tqdm(abstracts, disable=not self.verbose):
            abstract_embedding = self.embedding_model.invoke(abstract)
            cosine_sim = cosine_similarity(abstract_embedding, reserch_embedding)
            cosine_similarities.append(cosine_sim)
            abstract_embeddings.append(abstract_embedding)
        
        data_df['cosine_similarity'] = cosine_similarities
        data_df['abstract_embedding'] = abstract_embeddings
        # Filter the dataframe based on cosine similarity threshold
        filtered_df = data_df[data_df['cosine_similarity'] >= self.cosine_similarity_threshold]

        # Reset the index of the filtered dataframe
        filtered_df = filtered_df.reset_index(drop=True)

        # Update data_df with the filtered results
        data_df = filtered_df

        return data_df
    

    def rank_papers(self, data_df):
        """Evaluates remaining papers and ranks them with the generative model."""
        abstracts = list(data_df['abstract'])
        scores = []
        recommends = []
        rationale = []
        for abstract in tqdm(abstracts, disable=not self.verbose):
            messages = [{"role": "user", "content": research_prompt(self.research_interests, abstract)}]
            response = self.inference.invoke(messages=messages, system_prompt=RESEARCH_INTERESTS_SYSTEM_PROMPT)
            response_json = json_repair.loads(response)
            scores.append(int(response_json['score']))
            recommends.append(bool(response_json['recommend']))
            rationale.append(response_json['rationale'])
        
        data_df['score'] = scores
        data_df['recommend'] = recommends
        data_df['rationale'] = rationale
        # Sort the DataFrame by score in descending order
        data_df = data_df.sort_values(by='score', ascending=False)
        top_n_df = data_df.head(self.top_n)

        # Convert each row of the data_df to a Paper class and place them into a list
        papers = []
        for _, row in data_df.iterrows():
            paper = Paper(
                title=row['title'],
                abstract=row['abstract'],
                url=row['url_pdf'],
                date_run=TODAY.strftime('%Y-%m-%d'),
                date=row['date'].strftime('%Y-%m-%d'),
                score=row['score'],
                recommend=row['recommend'],
                rationale=row['rationale'],
                cosine_similarity=row['cosine_similarity'],
                embedding_model=self.embedding_model_name
            )
            papers.append(paper)
        
            self.papers_db.insert_papers(paper)
        return top_n_df
    
    def generate_newsletter(self, top_n_df):
        """Generates a newsletter from the ranked papers."""
        content = []
        for _, row in top_n_df.iterrows():
            content.append(f"{row['title']}: {row['abstract']}")
        content = "\n".join(content)
        content = newsletter_prompt(content, self.research_interests)
        newsletter_draft = self.inference.invoke(messages=[{"role": "user", "content": content}], system_prompt=NEWSLETTER_SYSTEM_PROMPT)
        newsletter_draft_json = json_repair.loads(newsletter_draft)
        newsletter_content = newsletter_draft_json['newsletter']
        
        newsletter = Newsletter(
            content=newsletter_content,
            start_date=self.start_date.strftime('%Y-%m-%d'),
            end_date=self.end_date.strftime('%Y-%m-%d'),
            date_sent=TODAY.strftime('%Y-%m-%d')
        )
        self.papers_db.insert_newsletter(newsletter)

        email_body = construct_email_body(newsletter_content, self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))
        self.communication.send_email(email_body)

    def run(self):
        """Runs the PaperPal system."""
        data_df = self.download_and_process_papers()
        top_n_df = self.rank_papers(data_df)
        self.generate_newsletter(top_n_df)
        

