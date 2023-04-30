import argparse
from communication import GmailCommunication
from database import DatabaseUtils
from inference import Inference
from processdata import ProcessData


def get_research_interests(filename):
    """Function to generate the research interests information.

    Args:
        filename (str): Filename of your text file with research interests.

    Returns:
        str: concatenated research interests by newline.
    """
    with open(filename, 'r') as f:
        interests = f.readlines()
    interests = '/n'.join(interests)
    return interests