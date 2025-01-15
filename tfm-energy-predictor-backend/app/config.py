class Config:
    SECRET_KEY = '_fill_your_own_secret_key_here_'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    URL_ORIGIN_CORS = 'http://localhost:4200'
    DATABASE_URL = "sqlite:///./test.db"
