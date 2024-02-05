from os import environ
from dotenv import load_dotenv
load_dotenv(verbose=True)


class Config:
    DATABASE_HOST = environ.get('DATABASE_HOST')
    DATABASE_USERNAME = environ.get('DATABASE_USERNAME')
    DATABASE_PASSWORD = environ.get('DATABASE_PASSWORD')
    DATABASE_PORT = environ.get('DATABASE_PORT')
    DATABASE_NAME = environ.get('DATABASE_NAME')
    OPENAI_API_KEY = environ.get('OPENAI_API_KEY')
