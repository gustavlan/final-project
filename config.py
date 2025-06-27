import os
from dotenv import load_dotenv

load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Basic Flask settings
    # SECRET_KEY must be provided through the environment for production and
    # development configurations.
    SECRET_KEY = os.environ.get('SECRET_KEY')
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
    # Static daily risk-free rate (optional, used if FRED_API_KEY is not set)
    RISK_FREE_RATE = (
        float(os.environ.get('RISK_FREE_RATE'))
        if os.environ.get('RISK_FREE_RATE') is not None
        else None
    )

    @classmethod
    def validate(cls):
        """Ensure a SECRET_KEY is provided for non-testing configs."""
        if not cls.TESTING and not cls.SECRET_KEY:
            raise RuntimeError(
                "SECRET_KEY environment variable must be set for production and development"
            )

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'test-secret-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    DEBUG = False
