import sqlite3
import datetime
import pandas as pd
import os

class DatabaseUtils:
    def __init__(self, mkdirs=True):
        if mkdirs:
               os.makedirs('database', exist_ok=True)  # Let's store our outputs here
        self.TABLE_NAME = 'papers_data'
        self.TABLE_COLUMNS = {
                'paper_url': 'TEXT',
                'arxiv_id': 'TEXT PRIMARY KEY',
                'title': 'TEXT',
                'abstract': 'TEXT',
                'url_abs': 'TEXT',
                'url_pdf': 'TEXT',
                'proceeding': 'TEXT',
                'authors': 'TEXT',
                'tasks': 'TEXT',
                'date': 'DATETIME',
                'methods': 'TEXT',
                'summary': 'TEXT',
                'recommended': 'TEXT',
                'why_recommended': 'TEXT'}


        self.DATABASE_FILENAME = 'database/paperpal_sqlite.db'
        self.CREATE_TABLE_QUERY = "CREATE TABLE IF NOT EXISTS {table} ({fields})"

        self.INSERT_QUERY = "INSERT INTO {table} ({columns}) VALUES ({values})"


    def create_connection(self, db_filename=None):
        """Function to initialize the connection to the db.  Note that this will create a file
        if one does not already exist

        Args:
            db_filename (str, optional): Path of the database file. Defaults to DATABASE_FILENAME.

        Returns:
            sqlite connection: connection object for the sqlite database.
        """
        if not db_filename:
            db_filename = self.DATABASE_FILENAME
        try:
            conn = sqlite3.connect(db_filename)
            return conn
        except Exception as e:
            print(e)


    def create_table(self, conn, table=None, name=None):
            """Function to conduct the intiial table creation for
            SQLite
            Args:
                conn (sqlite cxn): SQLite3 Connection
                table (dict): schema for the table to be created
                name (str): [description]
            """
            if not table:
                 table = self.TABLE_COLUMNS
            if not name:
                 name = self.TABLE_NAME
            create_table = self.CREATE_TABLE_QUERY

            columns = ['{0} {1}'.format(name, ctype) for name, ctype in table.items()]
            create = create_table.format(table=name, fields=", ".join(columns))

            try:
                conn.execute(create)
            except Exception as e:
                print


    def insert_data(self, conn, row_dict, table_name):
        """
        Inserts a list of dictionaries or a single dictionary into a SQLite database table as rows.

        Args:
            conn (sqlite cxn): SQLite3 Connection
            row_dict (dict): A list of dictionaries or a single dictionary to insert.
            table_name (str): The name of the SQLite database table to insert the rows into.

        """


        # Get the column names and values from the dictionary object.
        column_names = list(row_dict.keys())
        values = list(row_dict.values())

    # Create a SQL INSERT statement.
        insert_statement = 'INSERT INTO {} ({}) VALUES ({})'.format(
                            table_name,
                            ', '.join(column_names),
                            ', '.join(['?'] * len(column_names)))

        cursor = conn.cursor()

        # Execute the SQL INSERT statement.
        cursor.executemany(insert_statement, values)

        conn.commit()


