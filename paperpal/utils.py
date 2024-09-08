import datetime
import numpy as np


TODAY = datetime.date.today()

def get_n_days_ago(n_days):
    """
    Get the date n days ago from today.

    Args:
        n_days (int): Number of days to look back.

    Returns:
        datetime.date: The date n days ago.
    """
    return TODAY - datetime.timedelta(days=n_days)


def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        a (numpy.ndarray): First vector.
        b (numpy.ndarray): Second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
