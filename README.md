# PaperPal

PaperPal is a tool for sorting and analyzing research papers based on your personal research interests. It's designed to be accessible and customizable, allowing users to adapt it for their specific needs.

## Key Features

- Integration with various language models (Llama 3.1, OpenAI, Anthropic) for paper summarization and recommendation
- Support for using different models for different tasks (judging and newsletter generation)
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

# Run with custom parameters
python run_paperpal.py --n-days 14 --top-n 20 --model-name llama2
```

### Available Arguments

- `--research-interests-path`: Path to research interests file (default: "config/research_interests.txt")
- `--n-days`: Number of days to look back for papers (default: 7)
- `--top-n`: Number of top papers to return (default: 10)
- `--use-different-models`: Use different models for different tasks (default: True)
- `--model-type`: Type of model to use (default: "ollama")
- `--model-name`: Name of the model to use (default: "hermes3")
- `--orchestration-config`: Path to config for multiple models (default: "config/orchestration.json")
- `--embedding-model-name`: Name of the embedding model (default: "Alibaba-NLP/gte-base-en-v1.5")
- `--cosine-similarity-threshold`: Threshold for paper filtering (default: 0.5)
- `--temperature`: Temperature for text generation (default: 0.1)
- `--max-new-tokens`: Maximum new tokens to generate (default: 1024)
- `--receiver-address`: Email address for notifications (default: None)
- `--data-path`: Path to the database file (default: "data/papers.db")
- `--verbose`: Enable verbose output (default: True)

### Using Multiple Models

When `--use-different-models` is enabled, PaperPal can use different models for judging papers and generating newsletters. Configure this in `config/orchestration.json`:

```json
{
    "judge_model": {
        "model_type": "ollama",
        "model_name": "llama2",
        "max_new_tokens": 1024,
        "temperature": 0.1
    },
    "newsletter_model": {
        "model_type": "anthropic",
        "model_name": "claude-3-sonnet",
        "max_new_tokens": 2048,
        "temperature": 0.7
    }
}
```

## Configuration

- Model selection: Set the `MODEL_TYPE` environment variable to "ollama", "anthropic", or "openai"
- Email: Configure Gmail API access or modify `communication.py` for other email providers
- Research interests: Update `config/research_interests.txt` with your research interests

## Customization

- Prompts: Modify prompts in `paperpal/prompts.py` to adjust the AI's behavior
- Inference: Add or modify inference methods in `paperpal/inference.py` for different models
- Model orchestration: Configure different models for different tasks in `config/orchestration.json`

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
