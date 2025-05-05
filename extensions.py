# extentions.py to avoid circular imports
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from flask import session

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
# login_manager.init_app(app)

# Login manager configuration
login_manager.login_view = 'views.login'
login_manager.login_message = 'Please log in to access this page'
login_manager.login_message_category = 'info'