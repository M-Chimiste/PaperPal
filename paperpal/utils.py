import datetime
import numpy as np
import requests

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


def purge_ollama_cache(ollama_url, model_name):
    """
    Purge the cache for a specific Ollama model.

    This function sends a request to the Ollama API to clear the cache for the specified model,
    which can help free up system resources. It sends a generate request with keep_alive=0 to
    ensure the model is unloaded.

    Args:
        ollama_url (str): The base URL of the Ollama API server (e.g. "http://localhost:11434")
        model_name (str): The name of the model to purge from cache

    Returns:
        None
    """
    requests.post(f"{ollama_url}/api/generate", json={"model": model_name, "keep_alive": 0})