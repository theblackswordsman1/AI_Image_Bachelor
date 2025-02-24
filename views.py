from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from database import list_users, verify, delete_user_from_db, add_user
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db
from models import User

views = Blueprint('views', __name__)

@views.route("/")
def home():
    return render_template("index.html")

@views.route("/register")
def register():
    return render_template("register.html")

@views.route("/login")
def login():
    return render_template("login.html")


