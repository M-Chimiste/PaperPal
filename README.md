# PaperPal

PaperPal is a tool for sorting and analyzing research papers based on your personal research interests. It's designed to be accessible and customizable, allowing users to adapt it for their specific needs.

## Key Features

- Integration with various language models (Vicuna, OpenAI, Anthropic) for paper summarization and recommendation.
- Automated paper downloads from Papers with Code.
- Saving outputs to a SQLite database.
- Automated email notifications with research digests.
- Customizable research interests.

## Requirements

- A machine with good computational resources. CPU support is not explicitly provided, but contributions for CPU or MPS support are welcome.
- Python 3.7+
- PyTorch 2.4+
- CUDA 11.7+ (for GPU support)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/PaperPal.git
   cd PaperPal
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the project root and add your API keys:
     ```
     ANTHROPIC_API_KEY=your_anthropic_key
     OPENAI_API_KEY=your_openai_key
     GMAIL_SENDER_ADDRESS=your_gmail_address
     GMAIL_APP_PASSWORD=your_gmail_app_password
     ```

4. Configure Gmail:
   - To use Gmail for sending emails, you need to set up an application password. Follow [these instructions](https://support.google.com/mail/answer/185833?hl=en) to create an app password for Gmail.
   - Add your Gmail address and app password to the `.env` file as shown above.

## Usage

1. Update the `config/research_interests.txt` file with your research interests.

2. Run the script:
   ```
   python paperpal/paperpal.py --start_date "2023-04-24"
   ```

   Adjust the arguments as needed. Use `--help` to see all available options.

## Configuration

- Model selection: Set the `MODEL_TYPE` environment variable to "local", "anthropic", or "openai".
- Email: Configure Gmail API access or modify `communication.py` for other email providers.

## Customization

- Prompts: Modify prompts in `paperpal/prompts.py` to adjust the AI's behavior.
- Inference: Add or modify inference methods in `paperpal/inference.py` for different models.

## Database

PaperPal uses SQLite to store paper information and generated newsletters. The database schema can be found in `paperpal/data_handling.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgements

- [Papers with Code](https://paperswithcode.com/) for providing the research paper data.
- [Hugging Face](https://huggingface.co/) for transformer models and tokenizers.
