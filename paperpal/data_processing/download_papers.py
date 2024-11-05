import warnings
from pathlib import Path

import requests

class ArxivData:
    def __init__(self, url=None, arxiv_id=None) -> None:
        self.url = url
        self.arxiv_id = arxiv_id
        self.markdown_data = None
        if url:
            self.pdf_path = self.download_url()
        if arxiv_id:
            self.pdf_path = self.download_id()
        else:
            self.pdf_path = None
        if not url and not arxiv_id:
           
            warnings.warn("No URL or Arxiv ID provided. To download a PDF, please pass a URL or Arxiv ID as a parameter, or call the download_url or download_id methods manually.", UserWarning)
    

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