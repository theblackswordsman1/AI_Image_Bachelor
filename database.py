from extensions import db
from models import User
import hashlib
from werkzeug.security import generate_password_hash, check_password_hash

def list_users():
    return User.query.all()

def add_user(email, username, password):
    hashed_password = generate_password_hash(password)
    user = User(email=email, username=username, password=hashed_password)
    db.session.add(user)
    db.session.commit()
    return user

def verify(email, password):
    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        return user
    return None

def delete_user_from_db(user_id):
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        return True
    return False