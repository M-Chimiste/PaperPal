import re
from typing import Dict, Union, List
from pathlib import Path
from docling import DocumentConverter
import requests
import warnings


def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Parse a PDF file to markdown using the docling library.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The markdown content of the PDF.
    """
    converter = DocumentConverter()
    response = converter.converter.convert_single(pdf_path)
    markdown = response.render_as_markdown()
    return markdown


class MarkdownParser:
    """
    MarkdownParser is a class that parses markdown content into a hierarchical dictionary structure.

    Args:
        source (Union[str, Path]): The source of the markdown content, either as a string or a file path.

    Attributes:
        content (str): The raw markdown content.
        parsed_data (Dict): The parsed hierarchical representation of the markdown content.

    Methods:
        _load_content(source: Union[str, Path]) -> str:
            Loads the markdown content from a file or directly from a string.
        _parse_markdown() -> Dict:
            Parses the loaded markdown content into a hierarchical dictionary.
        get_parsed_data() -> Dict:
            Returns the parsed markdown data.
        find_sections(keyword: str) -> List[Dict]:
            Finds sections containing the given keyword.
    """
    def __init__(self, source: Union[str, Path]):
        self.content = self._load_content(source)
        self.parsed_data = self._parse_markdown()

    def _load_content(self, source: Union[str, Path]) -> str:
        if isinstance(source, Path):
            with open(source, 'r') as file:
                return file.read()
        return source

    def _parse_markdown(self) -> Dict:
        lines = self.content.split('\n')
        root = {}
        stack = [root]
        current_level = 0

        for line in lines:
            match = re.match(r'^(#+)\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                title = match.group(2)

                while level <= current_level:
                    stack.pop()
                    current_level -= 1

                new_section = {}
                stack[-1][title] = new_section
                stack.append(new_section)
                current_level = level
            else:
                content = stack[-1].get('content', '')
                stack[-1]['content'] = content + line + '\n'

        return root

    def get_parsed_data(self) -> Dict:
        return self.parsed_data

    def find_sections(self, keyword: str) -> List[Dict]:
        """
        Find sections that contain the given keyword (case-insensitive).
        Matches sections even if the keyword is part of a larger title.
        """
        results = []
        self._search_sections(self.parsed_data, keyword.lower(), results)
        return results

    def _search_sections(self, data: Dict, keyword: str, results: List[Dict], path: List[str] = []):
        for title, content in data.items():
            current_path = path + [title]
            if keyword in title.lower():
                results.append({
                    'path': current_path,
                    'content': content.get('content', ''),
                    'subsections': {k: v for k, v in content.items() if k != 'content'}
                })
            elif isinstance(content, dict):
                self._search_sections(content, keyword, results, current_path)


class ReferencesParser(MarkdownParser):
    def __init__(self, content: str):
        super().__init__(content)
        self.references = self._extract_references()

    def _extract_references(self) -> List[str]:
        """
        Extract references from the parsed markdown content.

        Returns:
            List[str]: A list of references found in the content.
        """
        references_section = self.find_sections("references")
        if not references_section:
            return []

        references_content = references_section[0].get('content', '')
        references = self._parse_references(references_content)
        return references

    def _parse_references(self, content: str) -> List[str]:
        """
        Parse the references content to extract individual references.

        Args:
            content (str): The content of the references section.

        Returns:
            List[str]: A list of individual references.
        """
        # Split the content by newlines to get individual references
        references = content.split('\n')
        # Filter out empty lines and strip leading/trailing whitespace
        references = [ref.strip() for ref in references if ref.strip()]
        return references

    def get_references(self) -> List[str]:
        """
        Get the list of extracted references.

        Returns:
            List[str]: A list of references.
        """
        return self.references


class ArxivData:
    def __init__(self, url=None, arxiv_id=None) -> None:
        self.url = url
        self.arxiv_id = arxiv_id
        
        if url:
            self.pdf_path = self.download_url()
        if arxiv_id:
            self.pdf_path = self.download_id()
        else:
            self.pdf_path = None
        if not url and not arxiv_id:
           
            warnings.warn("No URL or Arxiv ID provided. To download a PDF, please pass a URL or Arxiv ID as a parameter, or call the download_url or download_id methods manually.", UserWarning)

        self.markdown_data = self.extract_content()


    def download_url(self, url=None):
        """Method to download a pdf from a given url

        Args:
            url (str): The url to download the pdf from.

        Returns:
            str: The path to the downloaded pdf.
        """
        url = url or self.url
        response = requests.get(url)
        temp_pdf_name = url.split('/')[-1]
        with open(f'temp_data/{temp_pdf_name}', 'wb') as f:
            f.write(response.content)
        return f'temp_data/{temp_pdf_name}'
    

    def download_id(self, arxiv=None):
        """Method to download a pdf from a given arxiv id

        Args:
            arxiv (str): The arxiv id to download the pdf from.

        Returns:
            str: The path to the downloaded pdf.
        """
        arxiv = arxiv or self.arxiv_id
        url = f"https://arxiv.org/pdf/{arxiv}.pdf"
        response = requests.get(url)
        with open('temp_data/temp.pdf', 'wb') as f:
            f.write(response.content)
        return 'temp_data/temp.pdf'
    

    def extract_content(self):
        """Method to extract the content from the pdf"""
        markdown = parse_pdf_to_markdown(self.pdf_path)
        return markdown
