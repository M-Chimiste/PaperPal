import argparse
from paperpal import PaperPal

def parse_args():
    parser = argparse.ArgumentParser(description="Run PaperPal with custom configurations")
    
    # Add arguments with current defaults
    parser.add_argument("--research-interests-path", type=str, 
                       default="config/research_interests.txt",
                       help="Path to research interests file")
    
    parser.add_argument("--n-days", type=int, default=7,
                       help="Number of days to look back for papers")
    
    parser.add_argument("--top-n", type=int, default=5,
                       help="Number of top papers to return")
    
    parser.add_argument("--use-different-models", action="store_true", default=True,
                       help="Whether to use different models for different tasks")
    
    parser.add_argument("--model-type", type=str, default="ollama",
                       help="Type of model to use")
    
    parser.add_argument("--model-name", type=str, default="hermes3",
                       help="Name of the model to use")
    
    parser.add_argument("--orchestration-config", type=str,
                       default="config/orchestration.json",
                       help="Path to orchestration config file for multiple models")
    
    parser.add_argument("--embedding-model-name", type=str, 
                       default="Alibaba-NLP/gte-base-en-v1.5",
                       help="Name of the embedding model")
    
    parser.add_argument("--trust-remote-code", action="store_true", default=True,
                       help="Whether to trust remote code")
    
    parser.add_argument("--receiver-address", type=str, default=None,
                       help="Email address to receive notifications")
    
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                       help="Maximum number of new tokens to generate")
    
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for text generation")
    
    parser.add_argument("--cosine-similarity-threshold", type=float, default=0.5,
                       help="Threshold for cosine similarity")
    
    parser.add_argument("--db-saving", action="store_true", default=True,
                       help="Whether to save results to database")
    
    parser.add_argument("--data-path", type=str, default="data/papers.db",
                       help="Path to the database file")
    
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Whether to print verbose output")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    paperpal = PaperPal(
        research_interests_path=args.research_interests_path,
        n_days=args.n_days,
        top_n=args.top_n,
        use_different_models=args.use_different_models,
        model_type=args.model_type,
        model_name=args.model_name,
        orchestration_config=args.orchestration_config,
        embedding_model_name=args.embedding_model_name,
        trust_remote_code=args.trust_remote_code,
        receiver_address=args.receiver_address,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        cosine_similarity_threshold=args.cosine_similarity_threshold,
        db_saving=args.db_saving,
        data_path=args.data_path,
        verbose=args.verbose
    )
    
    paperpal.run()