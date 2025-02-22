import os
from dotenv import load_dotenv

load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Basic Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    DEBUG = False
    TESTING = False

    # Database configuration (SQLite example)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'final_project.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Logging configuration parameters
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')
    LOG_FILE = os.path.join(basedir, 'logs', 'final_project.log')
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    
    # FRED API Key (optional, can be set in environment)
    FRED_API_KEY = os.environ.get('FRED_API_KEY')

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    DEBUG = False
