# PaperPal

PaperPal is a tool for sorting and analyzing research papers based on your personal research interests. It's designed to be accessible and customizable, allowing users to adapt it for their specific needs.

## Key Features

- Integration with various language models (Llama 3.1, OpenAI, Anthropic) for paper summarization and recommendation
- Support for using different models for different tasks (judging and newsletter generation) based on configuration
- Automated paper downloads from Papers with Code
- Saving outputs to a SQLite database
- Automated email notifications with research digests
- Customizable research interests
- Embedding-based paper filtering with configurable similarity thresholds

## Requirements

- A machine with good computational resources if you are not using a LLM with an API. CPU and MPS are supported through Ollama. Check out the [Ollama](https://ollama.com/) website for more information.
- PyTorch 2.4+
- CUDA 11.7+ (for GPU support)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PaperPal.git
   cd PaperPal
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the project root and add your API keys:
     ```
     ANTHROPIC_API_KEY=your_anthropic_key
     OPENAI_API_KEY=your_openai_key
     GOOGLE_API_KEY=your_google_api_key
     GMAIL_SENDER_ADDRESS=your_gmail_address
     GMAIL_APP_PASSWORD=your_gmail_app_password
     OLLAMA_URL=http://localhost:11434
     ```

4. Configure Gmail:
   - To use Gmail for sending emails, you need to set up an application password. Follow [these instructions](https://support.google.com/mail/answer/185833?hl=en) to create an app password for Gmail.
   - Add your Gmail address and app password to the `.env` file as shown above.

## Usage

The main script to run PaperPal is `run_paperpal.py`. You can run it with default settings or customize various parameters:

```bash
# Run with default settings
python run_paperpal.py
```
### Running with Custom Parameters

```bash
python run_paperpal.py --n-days 14 --top-n 20 --model-name llama2
```

### Available Arguments

- `--research-interests-path`: Path to research interests file (default: "config/research_interests.txt")
- `--n-days`: Number of days to look back for papers (default: 7)
- `--top-n`: Number of top papers to return (default: 5)
- `--use-different-models`: Use different models for different tasks (default: True)
- `--model-type`: Type of model to use (default: "ollama")
- `--model-name`: Name of the model to use (default: "hermes3")
- `--orchestration-config`: Path to config for multiple models (default: "config/orchestration.json")
- `--embedding-model-name`: Name of the embedding model (default: "Alibaba-NLP/gte-base-en-v1.5")
- `--trust-remote-code`: Whether to trust remote code (default: True)
- `--receiver-address`: Email address for notifications (default: None)
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 1024)
- `--temperature`: Temperature for text generation (default: 0.1)
- `--cosine-similarity-threshold`: Threshold for cosine similarity (default: 0.5)
- `--db-saving`: Whether to save results to database (default: True)
- `--data-path`: Path to the database file (default: "data/papers.db")
- `--verbose`: Enable verbose output (default: True)
- `--start-date`: Start date for paper retrieval (default: None)
- `--end-date`: End date for paper retrieval (default: None)

## Using Multiple Models

PaperPal supports the use of multiple models for different tasks such as judging papers and generating newsletters. This is configured via the `config/orchestration.json` file and leveraged by the `PaperPal` class.

#### 1. **Configure `orchestration.json`**

Define different models for specific tasks in the `config/orchestration.json` file.

```json:config/orchestration.json
{
    "judge_model": {
        "model_name": "hermes3",
        "model_type": "ollama",
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "num_ctx": 4096
    },
    "newsletter_model": {
        "model_name": "hermes3",
        "model_type": "ollama",
        "max_new_tokens": 4096,
        "temperature": 0.1,
        "num_ctx": 131072
    },
    "content_extraction_model": {
        "model_name": "hermes3",
        "model_type": "ollama",
        "max_new_tokens": 4096,
        "temperature": 0.1,
        "num_ctx": 131072
    },
    "newsletter_draft_model": {
        "model_name": "hermes3",
        "model_type": "ollama",
        "max_new_tokens": 4096,
        "temperature": 0.1,
        "num_ctx": 131072
    },
    "newsletter_revision_model": {
        "model_name": "gemini-1.5-flash",
        "model_type": "gemini",
        "max_new_tokens": 4096,
        "temperature": 0.1,
        "num_ctx": 131072
    }
}
```

## Configuration

- **Model Selection**: Set the `MODEL_TYPE` environment variable to "ollama", "anthropic", "openai", or "gemini" to choose the default model type.
- **Orchestration**: Configure different models for different tasks in `config/orchestration.json`.
- **Email**: Configure Gmail API access or modify `communication.py` for other email providers.
- **Research Interests**: Update `config/research_interests.txt` with your research interests.

## Customization

- **Prompts**: Modify prompts in `paperpal/prompts.py` to adjust the AI's behavior.
- **Inference**: Add or modify inference methods in `paperpal/inference.py` for different models.
- **Model Orchestration**: Configure different models for different tasks in `config/orchestration.json`.

## Database

PaperPal uses SQLite to store paper information and generated newsletters. The database schema can be found in `paperpal/data_handling.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgements

- [Papers with Code](https://paperswithcode.com/) for providing the research paper data
- [Hugging Face](https://huggingface.co/) for transformer models and tokenizers
- [Ollama](https://ollama.com/) for local model support
