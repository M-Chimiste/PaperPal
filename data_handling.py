import os
import sqlite3
from pathlib import Path
from pydantic import BaseModel, field_validator
from contextlib import contextmanager

class Paper(BaseModel):
    title: str
    abstract: str
    authors: str
    date: str
    date_run: str
    score: float
    rationale: str
    recommended: bool
    cosine_similarity: float
    url: str

    @field_validator('score')
    @classmethod
    def score_range(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('Score must be between 0 and 10')
        return v

class PaperDatabase:
    """
    PaperDatabase class for handling paper data storage and retrieval.

    This class provides methods to interact with a SQLite database for storing
    and retrieving paper information. It includes functionality to insert new
    papers, retrieve papers based on various criteria, and manage the database
    connection.

    Attributes:
        db_path (Path): The path to the SQLite database file.
        conn (sqlite3.Connection): The database connection object.

    Methods:
        __init__(db_path: str = "papers.db"):
            Initialize the PaperDatabase instance.
        
        _ensure_path_exists():
            Ensure the database directory exists.
        
        get_cursor():
            Context manager that yields a database cursor and handles transactions.
        
        _create_table():
            Create the papers table if it doesn't exist.
        
        insert_paper(paper: Paper):
            Insert a new paper into the database.
        
        close():
            Close the database connection.
    """
    def __init__(self, db_path: str = "papers.db"):
        self.db_path = Path(db_path)
        self._ensure_path_exists()
        self.conn = self._create_connection()
        self._create_table()

    def _ensure_path_exists(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_cursor(self):
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def _create_table(self):
        with self.get_cursor() as cursor:
            cursor.execute('''CREATE TABLE IF NOT EXISTS papers
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               title TEXT NOT NULL,
                               abstract TEXT NOT NULL,
                               authors TEXT NOT NULL,
                               date TEXT NOT NULL,
                               date_run TEXT NOT NULL,
                               score REAL NOT NULL,
                               rationale TEXT NOT NULL,
                               recommended BOOLEAN NOT NULL,
                               cosine_similarity REAL NOT NULL,
                               url TEXT NOT NULL
                            )''')

    def insert_paper(self, paper: Paper):
        with self.get_cursor() as cursor:
            cursor.execute('''INSERT INTO papers (title, abstract, authors, date, date_run, score, rationale, recommended, cosine_similarity, url)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (paper.title, paper.abstract, paper.authors, paper.date, paper.date_run, 
                            paper.score, paper.rationale, paper.recommended, paper.cosine_similarity, paper.url))

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None
