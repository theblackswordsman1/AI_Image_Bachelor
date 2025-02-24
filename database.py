from extensions import db
from models import User
import hashlib

def list_users():
    return User.query.all()

def add_user(email, username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user = User(email=email, username=username, password=hashed_password)
    db.session.add(user)
    db.session.commit()
    return user

def verify(email, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user = User.query.filter_by(email=email, password=hashed_password).first()
    return user

def delete_user_from_db(user_id):
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        return True
    return False