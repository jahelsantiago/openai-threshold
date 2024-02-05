import sys
import psycopg2
from config import Config
from psycopg2.extras import DictCursor



class Database:
    """PostgreSQL Database class."""
    def __init__(self, config):
        self.host = config.DATABASE_HOST
        self.username = config.DATABASE_USERNAME
        self.password = config.DATABASE_PASSWORD
        self.port = config.DATABASE_PORT
        self.dbname = config.DATABASE_NAME
        self.conn = None

    def connect(self):
        """Connect to a Postgres database."""
        if self.conn is None:
            try:
                self.conn = psycopg2.connect(
                    host=self.host,
                    user=self.username,
                    password=self.password,
                    port=self.port,
                    dbname=self.dbname)
            except psycopg2.DatabaseError as e:
                print("error::", e)
                sys.exit()
            finally:
                print('Connection opened successfully.')

    def select_rows(self, query):
        """Run a SQL query to select rows from table."""
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(query)
            records = [row for row in cur.fetchall()]
            cur.close()
            return records

    def select_rows_dict_cursor(self, query):
        """Run a SQL query to select rows from table and return dictionarys."""
        self.connect()
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query)
            return cur.fetchall()


print("Starting")
db = Database(Config)
results = db.select_rows_dict_cursor('SELECT pk, text_raw FROM nlp.chat_gpt_company_embedding limit 10;')



