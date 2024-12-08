# Standard library imports
import json
import os
import datetime
# Third-party imports
from dotenv import load_dotenv
import json_repair
from tqdm import tqdm
from docling.document_converter import DocumentConverter

# Local application imports
from .communication import GmailCommunication, construct_email_body
from .data_processing import ProcessData, PaperDatabase, Paper, Newsletter
from .data_processing.data_handling import PaperDatabase, Paper, Newsletter
from .llm import SentenceTransformerInference
from .pdf import MarkdownParser, ArxivData, parse_pdf_to_markdown
from .prompt import (
    NEWSLETTER_SYSTEM_PROMPT,
    RESEARCH_INTERESTS_SYSTEM_PROMPT,
    SYSTEM_CONTENT_EXTRACTION_SUMMARY,
    general_summary_prompt,
    research_prompt,
    newsletter_context_prompt,
    newsletter_final_prompt,
    newsletter_intro_prompt,
    SummaryPromptData,
    ResearchInterestsPromptData,
    NewsletterPromptData
)
from .utils import cosine_similarity, get_n_days_ago, TODAY, purge_ollama_cache

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
GMAIL_SENDER_ADDRESS = os.getenv("GMAIL_SENDER_ADDRESS", None)
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", None)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")


class PaperPal:
    def __init__(self,
                 research_interests_path="config/research_interests.txt",
                 n_days=7,
                 top_n=5,
                 start_date=None,
                 end_date=None,
                 use_different_models=True,
                 model_type="ollama",
                 model_name="hermes3",
                 orchestration_config="config/orchestration.json",
                 embedding_model_name="Alibaba-NLP/gte-base-en-v1.5",
                 trust_remote_code=True,
                 receiver_address=None,
                 max_new_tokens=1024,
                 temperature=0.1,
                 cosine_similarity_threshold=0.5,
                 db_saving=True,
                 data_path="data/papers.db",
                 verbose=True):
        self.verbose = verbose
        self.research_interests_path = research_interests_path
        if start_date is None and end_date is None:
            self.start_date = get_n_days_ago(n_days)
            self.end_date = TODAY
        else:
            def try_parse_date(date_str):
                if not date_str:
                    return None
                formats = ["%Y-%m-%d", "%m-%d-%Y"]
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt).date()
                    except ValueError:
                        continue
                raise ValueError("Dates must be in YYYY-MM-DD or MM-DD-YYYY format")

            self.start_date = try_parse_date(start_date) or get_n_days_ago(n_days)
            self.end_date = try_parse_date(end_date) or TODAY
        self.start_date = get_n_days_ago(n_days)
        self.end_date = TODAY
        self.use_different_models = use_different_models
        self.top_n = top_n
        self.model_type = model_type
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.receiver_address = receiver_address
        self.communication = GmailCommunication(sender_address=GMAIL_SENDER_ADDRESS,
                                               app_password=GMAIL_APP_PASSWORD,
                                               receiver_address=receiver_address)
        self.papers_db = PaperDatabase(data_path)
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformerInference(embedding_model_name, trust_remote_code=trust_remote_code)
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.db_saving = db_saving
        # Load research interests
        try:
            with open(self.research_interests_path, 'r') as file:
                self.research_interests = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"The research interests file at {self.research_interests_path} could not be found. Please check the path and try again.")
        except IOError:
            raise IOError(f"There was an error reading the file at {self.research_interests_path}. Please check the file permissions and try again.")
        # Load inference model/s
        if not use_different_models:
            self.inference = self._load_inference_model(self.model_type, model_name, max_new_tokens, temperature)
        
        if use_different_models:
            with open(orchestration_config, 'r') as file:
                self.orchestration_config = json.load(file)
            self.judge_model_config = self.orchestration_config['judge_model']
            self.newsletter_model_config = self.orchestration_config['newsletter_model']
            self.content_extraction_model_config = self.orchestration_config['content_extraction_model']
            self.newsletter_draft_model_config = self.orchestration_config['newsletter_draft_model']
            self.newsletter_revision_model_config = self.orchestration_config['newsletter_revision_model']
            self.judge_inference = self._load_inference_model(self.judge_model_config['model_type'],
                                                                self.judge_model_config['model_name'],
                                                                self.judge_model_config['max_new_tokens'],
                                                                self.judge_model_config['temperature'],
                                                                self.judge_model_config.get('num_ctx', None))
            self.newsletter_inference = self._load_inference_model(self.newsletter_model_config['model_type'],
                                                                    self.newsletter_model_config['model_name'],
                                                                    self.newsletter_model_config['max_new_tokens'],
                                                                    self.newsletter_model_config['temperature'],
                                                                    self.newsletter_model_config.get('num_ctx', None))
            self.content_extraction_inference = self._load_inference_model(self.content_extraction_model_config['model_type'],
                                                                    self.content_extraction_model_config['model_name'],
                                                                    self.content_extraction_model_config['max_new_tokens'],
                                                                    self.content_extraction_model_config['temperature'],
                                                                    self.content_extraction_model_config.get('num_ctx', None))
            self.newsletter_draft_inference = self._load_inference_model(self.newsletter_draft_model_config['model_type'],
                                                                    self.newsletter_draft_model_config['model_name'],
                                                                    self.newsletter_draft_model_config['max_new_tokens'],
                                                                    self.newsletter_draft_model_config['temperature'],
                                                                    self.newsletter_draft_model_config.get('num_ctx', None))
            self.newsletter_revision_inference = self._load_inference_model(self.newsletter_revision_model_config['model_type'],
                                                                    self.newsletter_revision_model_config['model_name'],
                                                                    self.newsletter_revision_model_config['max_new_tokens'],
                                                                    self.newsletter_revision_model_config['temperature'],
                                                                    self.newsletter_revision_model_config.get('num_ctx', None))

    def _load_inference_model(self, model_type, model_name, max_new_tokens, temperature, num_ctx=None):
        """Load the appropriate inference model based on model type.
        
        Args:
            model_type (str): Type of model to load ('anthropic', 'openai', or 'ollama')
            model_name (str): Name of the specific model to load
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature parameter for generation
            
        Returns:
            The loaded inference model object
            
        Raises:
            ValueError: If model_type is invalid or required API keys are missing
        """
        # Check environment variable override
        model_type = os.getenv("MODEL_TYPE") or model_type
        
        if model_type == "anthropic":
            if ANTHROPIC_API_KEY is None:
                raise ValueError("Anthropic API key is not set. Please check your .env file and ensure ANTHROPIC_API_KEY is properly configured.")
            from .llm.inference import AnthropicInference
            return AnthropicInference(model_name, max_new_tokens, temperature)
            
        elif model_type == "openai":
            if OPENAI_API_KEY is None:
                raise ValueError("OpenAI API key is not set. Please check your .env file and ensure OPENAI_API_KEY is properly configured.")
            from .llm.inference import OpenAIInference
            return OpenAIInference(model_name, max_new_tokens, temperature)

        elif model_type == "gemini":
            if GOOGLE_API_KEY is None:
                raise ValueError("Google API key is not set. Please check your .env file and ensure GOOGLE_API_KEY is properly configured.")
            from .llm.inference import GeminiInference
            return GeminiInference(model_name, max_new_tokens, temperature)

        elif model_type == "ollama":
            from .llm.inference import OllamaInference
            kwargs = {
                'model_name': model_name,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'url': OLLAMA_URL
            }
            if num_ctx is not None:
                kwargs['num_ctx'] = num_ctx
            return OllamaInference(**kwargs)
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of 'local', 'anthropic', 'openai', or 'ollama'.")
        
        
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
        related = []
        rationale = []
        for abstract in tqdm(abstracts, disable=not self.verbose):
            messages = [{"role": "user", "content": research_prompt(self.research_interests, abstract)}]
            if not self.use_different_models:
                response = self.inference.invoke(messages=messages, system_prompt=RESEARCH_INTERESTS_SYSTEM_PROMPT, schema=ResearchInterestsPromptData)
            else:
                response = self.judge_inference.invoke(messages=messages, system_prompt=RESEARCH_INTERESTS_SYSTEM_PROMPT, schema=ResearchInterestsPromptData)
            response_json = json_repair.loads(response)
            scores.append(int(response_json['score']))
            related.append(bool(response_json['related']))
            rationale.append(response_json['rationale'])
        
        data_df['score'] = scores
        data_df['related'] = related
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
                related=row['related'],
                rationale=row['rationale'],
                cosine_similarity=row['cosine_similarity'],
                embedding_model=self.embedding_model_name
            )
            papers.append(paper)
            if self.db_saving:
                self.papers_db.insert_paper(paper)
        return top_n_df
    

    def generate_newsletter(self, top_n_df):
        """Generates a newsletter from the ranked papers."""
        # content = []
        sections = []
        urls_and_titles = []
        converter = DocumentConverter()
        total_rows = len(top_n_df)
        for i, (_, row) in enumerate(tqdm(top_n_df.iterrows(), total=total_rows, desc="Generating newsletter sections", disable=not self.verbose)):
            
           
            response = converter.convert(row['url_pdf'])
            markdown = response.document.export_to_markdown()
            messages = [{"role": "user", "content": general_summary_prompt(markdown)}]
            if not self.use_different_models:
                response = self.inference.invoke(messages=messages, system_prompt=SYSTEM_CONTENT_EXTRACTION_SUMMARY, schema=SummaryPromptData)
            else:
                response = self.content_extraction_inference.invoke(messages=messages, system_prompt=SYSTEM_CONTENT_EXTRACTION_SUMMARY, schema=SummaryPromptData)
            response_json = json_repair.loads(response)
            try:
                summarized_paper = response_json['content']
            except:
                summarized_paper = response

            context = f"Title: {row['title']}\nAbstract: {row['abstract']}\nRationale: {row['rationale']}\nSummary: {summarized_paper}"
            messages = [{"role": "user", "content": newsletter_context_prompt(self.research_interests, context)}]
            
            if not self.use_different_models:
                response = self.inference.invoke(messages=messages, system_prompt=NEWSLETTER_SYSTEM_PROMPT, schema=NewsletterPromptData)
            else:
                response = self.newsletter_inference.invoke(messages=messages, system_prompt=NEWSLETTER_SYSTEM_PROMPT, schema=NewsletterPromptData)
            
            response_json = json_repair.loads(response)
            draft = f"## {row['title']}\n\n{response_json['draft']}"
            sections.append(draft)
            urls_and_titles.append(f"{row['title']}: {row['url_pdf']}")
        # Format urls and titles as numbered markdown list
        urls_and_titles = "\n".join(f"{i+1}. {title}" for i, title in enumerate(urls_and_titles))
        # urls_and_titles = "\n".join(urls_and_titles)
        sections = "\n".join(sections)
        intro_prompt = newsletter_intro_prompt(sections)
        if not self.use_different_models:
            newsletter_intro = self.inference.invoke(messages=[{"role": "user", "content": intro_prompt}], system_prompt=NEWSLETTER_SYSTEM_PROMPT, schema=NewsletterPromptData)
        else:
            newsletter_intro = self.newsletter_draft_inference.invoke(messages=[{"role": "user", "content": intro_prompt}], system_prompt=NEWSLETTER_SYSTEM_PROMPT, schema=NewsletterPromptData)
        try:
            newsletter_intro_json = json_repair.loads(newsletter_intro)
            newsletter_intro = newsletter_intro_json['draft']
        except:
            newsletter_intro = newsletter_intro
        
        newsletter_content = f"{newsletter_intro}\n{sections}"
        messages = [{"role": "user", "content": newsletter_final_prompt(newsletter_content)}]
        if not self.use_different_models:
            newsletter_final = self.inference.invoke(messages=messages, system_prompt=NEWSLETTER_SYSTEM_PROMPT, schema=NewsletterPromptData)
        else:
            newsletter_final = self.newsletter_revision_inference.invoke(messages=messages, system_prompt=NEWSLETTER_SYSTEM_PROMPT, schema=NewsletterPromptData)
       
        newsletter_final_json = json_repair.loads(newsletter_final)
        newsletter_content = newsletter_final_json['draft']
        
        newsletter = Newsletter(
            content=newsletter_content,
            start_date=self.start_date.strftime('%Y-%m-%d'),
            end_date=self.end_date.strftime('%Y-%m-%d'),
            date_sent=TODAY.strftime('%Y-%m-%d')
        )
        if self.db_saving:  
            self.papers_db.insert_newsletter(newsletter)

        email_body = construct_email_body(newsletter_content, self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'), urls_and_titles)
        self.communication.compose_message(email_body, self.start_date, self.end_date)
        self.communication.send_email()


    def run(self):
        """Runs the PaperPal system."""
        data_df = self.download_and_process_papers()
        top_n_df = self.rank_papers(data_df)
        self.generate_newsletter(top_n_df)
        if self.model_type == "ollama":
            purge_ollama_cache(OLLAMA_URL, self.model_name)
        
if __name__ == "__main__":
    paperpal = PaperPal(
                 research_interests_path="config/research_interests.txt",
                 n_days=7,
                 top_n=10,
                 model_type="ollama",
                 model_name="hermes3",
                 embedding_model_name="Alibaba-NLP/gte-base-en-v1.5",
                 trust_remote_code=True,
                 receiver_address=None,
                 max_new_tokens=1024,
                 temperature=0.1,
                 cosine_similarity_threshold=0.5,
                 db_saving=True,
                 data_path="data/papers.db",
                 verbose=True)
    paperpal.run()