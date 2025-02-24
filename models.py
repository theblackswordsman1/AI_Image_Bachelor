from extensions import db
from flask_login import UserMixin
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(50), unique=True)
    username = db.Column(db.String(20), unique=True)
    password = db.Column(db.String(50))

class RegisterForm(FlaskForm):
    email = StringField("Email", validators=[InputRequired(), Length(min=4, max=50)])
    username = StringField("Username", validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=4, max=50)])
    submit = SubmitField("Register")

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError("Username already registered")

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[InputRequired(), Length(min=4, max=50)])
    username = StringField("Username", validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=4, max=50)])
    submit = SubmitField("Login")