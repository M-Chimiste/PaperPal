# Copyright 2023 M Chimiste

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "0.0.5"
from .paperpal import PaperPal
from .prompt import (research_interests_prompt,
                     newsletter_prompt,
                     research_prompt,
                     RESEARCH_INTERESTS_SYSTEM_PROMPT,
                     NEWSLETTER_SYSTEM_PROMPT)
from .llm import (LocalCudaInference, 
                   AnthropicInference, 
                   OpenAIInference, 
                   SentenceTransformerInference, 
                   OllamaInference)
from .utils import (cosine_similarity, 
                    get_n_days_ago, 
                    TODAY, 
                    purge_ollama_cache)
from .communication import construct_email_body, GmailCommunication
from .data_processing import ProcessData, PaperDatabase, Paper, Newsletter
from .pdf import parse_pdf_to_markdown, MarkdownParser, ReferencesParser
