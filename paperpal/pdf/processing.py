import os
import logging
from pathlib import Path
import re

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import spacy
from spacy_layout import spaCyLayout
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

load_dotenv()
class DoclingDocProcessor:
    """
    A class to process documents and export tables and figures.
    Attributes:
        doc_path (Path): The path to the document to be processed.
        export_tables (bool): Flag to indicate whether to export tables.
        export_figures (bool): Flag to indicate whether to export figures.
        verbose (bool): Flag to indicate whether to display progress information.
        table_format (str): The format in which to export tables ('csv', 'html', 'md').
        output_dir (Path): The directory where the output files will be saved.
        converter (DocumentConverter): The document converter instance.
        processed_doc: The processed document.
    Methods:
        __init__(doc_path, output_dir=None, table_format='csv', export_tables=True, export_figures=True, verbose=False):
            Initializes the DocProcessor with the given parameters.
        _process_doc(doc_path):
            Processes the document and returns the processed document.
        export_tables_method(output_dir):
        export_figures_method(output_dir):
    """

    def __init__(self, 
                 output_dir=None,
                 table_format='csv',
                 export_tables=True, 
                 export_figures=True,
                 save_text=False,
                 remove_md_image_tags=False,
                 verbose=False,
                 image_resolution_scale=2.0,
                 table_speed='accurate'):
        
        self.export_tables = export_tables
        self.export_figures = export_figures
        self.save_text = save_text
        self.remove_md_image_tags = remove_md_image_tags
        self.verbose = verbose
        self.output_dir = output_dir

        if table_format not in ['csv', 'html', 'md']:
            raise ValueError("table_format must be either 'csv', 'html', or 'md'")
        if table_speed not in ['accurate', 'fast']:
            raise ValueError("table_speed must be either 'accurate' or 'fast'")
        self.table_format = table_format

        self.output_dir = output_dir
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.images_scale = image_resolution_scale
        if table_speed == 'accurate':
            self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        if table_speed == 'fast':
            self.pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend, pipeline_options=self.pipeline_options
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline, 
                ),
            }
        )
        self.tables = None
        self.figures = None
        

    def export_markdown(self, converted_data):
        """
        Exports the converted data to a markdown format.

        Args:
            converted_data: The data that has been converted and needs to be exported.

        Returns:
            str: The markdown representation of the converted document.
        """
        data = converted_data.document.export_to_markdown()
        # Remove instances of 2+ newlines using regular expression
        data = re.sub(r'\n{2,}', '\n', data)
        if self.remove_md_image_tags:
            data = data.replace('<!-- image -->', '')
        return data
    
    def _process_doc(self, doc_path):
        return self.converter.convert(doc_path)
    

    def process_document(self, doc_path):
        """
        Processes a document and optionally exports tables and figures.
        Args:
            doc_path (str): The path to the document to be processed.
        Returns:
            dict: A dictionary containing the processed data, tables, and figures.
              The dictionary has the following keys:
              - "processed_data": The data processed from the document.
              - "tables": The exported tables if export_tables is True, otherwise None.
              - "figures": The exported figures if export_figures is True, otherwise None.
        """

        processed_data = self._process_doc(doc_path)
        markdown_data = self.export_markdown(processed_data)
        output_dir = self.output_dir or self.get_default_output_dir(doc_path)
        if self.save_text:
            with open(f"{output_dir}/text.md", "w") as f:
                f.write(markdown_data)
        if self.export_tables:
            tables = self.export_tables_method(output_dir, processed_data)
        else:
            tables = None
        if self.export_figures:
            figures = self.export_figures_method(output_dir, processed_data)
        else:
            figures = None
        if self.save_text:
            with open(f"{output_dir}/text.md", "w") as f:
                f.write(markdown_data)
        return {"processed_data": markdown_data, "tables": tables, "figures": figures}


    def get_default_output_dir(self, doc_path):
        """
        Generates and returns the default output directory for a given document path.

        This method takes the path to a document, extracts the filename, and creates
        a directory named after the filename (without extension) inside an 'output' folder.
        If the directory already exists, it will not raise an error.
        Args:
            str: The document path.
        Returns:
            str: The path to the default output directory.
        """
        base_path = self.output_dir or "output"
        filename = os.path.basename(doc_path)
        foldername = filename.split('.')[0]
        output_dir = f'{base_path}/{foldername}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir


    def export_tables_method(self, output_dir, processed_doc):
        """
        Exports tables from the processed document to the specified output directory in the desired format.
        Args:
            output_dir (str): The directory where the tables will be saved.
        Raises:
            ValueError: If the table format is not supported.
        Notes:
            The supported table formats are 'csv', 'html', and 'md'. The method uses the `tqdm` library to display a progress bar if `self.verbose` is True.
        """
        table_format = self.table_format
        table_filenames = []
        for table_idx, table in tqdm(enumerate(processed_doc.document.tables), disable=not self.verbose):
            table_df: pd.DataFrame = table.export_to_dataframe()
            table_filename = f"{output_dir}/table_{table_idx+1}.{table_format}"
            table_filenames.append(table_filename)
            if table_format == 'csv':
                table_df.to_csv(table_filename, index=False)
            elif table_format == 'html':
                with open(table_filename, 'w') as f:
                    f.write(table.export_to_html())
            elif table_format == 'md':
                table_df.to_markdown(table_filename, index=False)

        return table_filenames


    def export_figures_method(self, output_dir, processed_doc):
        """
        Extracts and saves all figures from the processed document to the specified output directory.
        Args:
            output_dir (str): The directory where the extracted figures will be saved.
        Returns:
            list: A list of filenames for the saved figures.
        """
        document = processed_doc.document
        picture_counter = 0
        picture_filenames = []
        for element, _level in document.iterate_items():
            if isinstance(element, PictureItem):
                try:
                    picture_counter += 1
                    element_image_filename = Path(f"{output_dir}/picture-{picture_counter}.png")
                    picture_filenames.append(element_image_filename)
                    with element_image_filename.open("wb") as fp:
                        element.get_image(document).save(fp, "PNG")
                except AttributeError as e:
                    logging.error(f"Error saving picture: {e}")
                    continue
        return picture_filenames
    

class SpacyLayoutDocProcessor:
    """
    A class to process documents and extract layout information using spaCy.
    Attributes:
        doc_path (Path): The path to the document to be processed.
        verbose (bool): Flag to indicate whether to display progress information.
        nlp: The spaCy model instance.
        processed_doc: The processed document.
    Methods:
        __init__(doc_path, verbose=False):
            Initializes the SpacyLayoutProcessor with the given parameters.
        _process_doc(doc_path):
            Processes the document and returns the processed document.
        extract_layout_info():
            Extracts layout information from the processed document.
    """

    def __init__(self,
                language="en",
                table_format='csv',
                verbose=False,
                do_export_tables=True, 
                do_export_figures=True,
                save_text=False,
                image_resolution_scale=2.0,
                output_dir=None,
                remove_md_image_tags=False):
        
        if table_format not in ['csv', 'html', 'md']:
            raise ValueError("table_format must be either 'csv', 'html', or 'md'")
        self.table_format = table_format
        self.verbose = verbose
        self.export_tables = export_tables
        self.export_figures = export_figures
        self.save_text = save_text
        self.output_dir = output_dir
        self.remove_md_image_tags = remove_md_image_tags
        self.nlp = spacy.blank(language)
        self.converter = spaCyLayout(self.nlp)

        if export_figures:
            self.pipeline_options = PdfPipelineOptions()
            self.pipeline_options.generate_picture_images = True
            self.pipeline_options.images_scale = image_resolution_scale
            self.docling_converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend, pipeline_options=self.pipeline_options
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline, 
                ),
            }
        )

        
    def _process_doc(self, doc_path):
        return self.converter(doc_path)
    

    def _docling_process_doc(self, doc_path):
        return self.docling_converter.convert(doc_path)
    
    def process_document(self, doc_path):
        output_dir = self.get_default_output_dir(doc_path)
        processed_doc = self._process_doc(doc_path)
        markdown_data = self.export_markdown(processed_doc)
        if self.save_text:
            with open(f"{output_dir}/text.md", "w") as f:
                f.write(markdown_data)
        if self.export_tables:
            tables = self.export_tables_method(output_dir, processed_doc)
        else:
            tables = None

        if self.export_figures:
            figure_data = self._docling_process_doc(doc_path)
            figures = self.export_figures_method(output_dir, figure_data)
        else:
            figures = None
        return {"processed_data": markdown_data, "tables": tables, "figures": figures}


    def get_default_output_dir(self, doc_path):
        """
        Generates and returns the default output directory for a given document path.

        This method takes the path to a document, extracts the filename, and creates
        a directory named after the filename (without extension) inside an 'output' folder.
        If the directory already exists, it will not raise an error.
        Args:
            str: The document path.
        Returns:
            str: The path to the default output directory.
        """
        base_path = self.output_dir or "output"
        filename = os.path.basename(doc_path)
        foldername = filename.split('.')[0]
        output_dir = '{base_path}/{foldername}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir


    def export_markdown(self, document):
        """
        Exports the markdown content of a document after processing.
        Args:
            document: The document object containing markdown content.
        Returns:
            str: The processed markdown content with optional image tags removed.
        """

        data = document._.markdown
        data = re.sub(r'\n{2,}', '\n', data)
        if self.remove_md_image_tags:
            data = data.replace('<!-- image -->', '')
        return data
    

    def export_tables_method(self, output_dir, document):
        """
        Exports tables from a document to the specified output directory in the specified format.
        Args:
            output_dir (str): The directory where the tables will be saved.
            document (Document): The document object containing tables to be exported.
        Returns:
            list: A list of filenames for the exported tables.
        Raises:
            ValueError: If the specified table format is not supported.
        Supported formats:
            - 'csv': Exports tables as CSV files.
            - 'html': Exports tables as HTML files.
            - 'md': Exports tables as Markdown files.
        """
        table_format = self.table_format
        tables = document._.tables
        table_filenames = []
        for table_idx, table in tqdm(enumerate(tables), disable=not self.verbose):
            table_df = table._.data
            table_filename = f"{output_dir}/table_{table_idx+1}.{table_format}"
            table_filenames.append(table_filename)
            if table_format == 'csv':
                table_df.to_csv(table_filename, index=False)
            elif table_format == 'html':
                with open(table_filename, 'w') as f:
                    f.write(table.export_to_html())
            elif table_format == 'md':
                table_df.to_markdown(table_filename, index=False)
        return table_filenames
    

    def export_figures_method(self, output_dir, processed_doc):
        """
        Extracts and saves all figures from the processed document to the specified output directory.
        Args:
            output_dir (str): The directory where the extracted figures will be saved.
        Returns:
            list: A list of filenames for the saved figures.
        """
        document = processed_doc.document
        picture_counter = 0
        picture_filenames = []
        for element, _level in document.iterate_items():
            if isinstance(element, PictureItem):
                try:
                    picture_counter += 1
                    element_image_filename = Path(f"{output_dir}/picture-{picture_counter}.png")
                    picture_filenames.append(element_image_filename)
                    with element_image_filename.open("wb") as fp:
                        element.get_image(document).save(fp, "PNG")
                except AttributeError as e:
                    logging.error(f"Error saving picture: {e}")
                    continue
        return picture_filenames    
