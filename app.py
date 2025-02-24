from flask import Flask
from extensions import db, migrate, login_manager

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
    app.config['SECRET_KEY'] = "fdsafasd"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    # Import models
    from models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Blueprints
    from views import views
    app.register_blueprint(views)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")