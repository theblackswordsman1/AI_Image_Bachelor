from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from database import list_users, verify, delete_user_from_db, add_user
from flask_login import login_user, logout_user, login_required, current_user, login_manager
from flask_wtf import FlaskForm
from extensions import db
from models import User, RegisterForm, LoginForm

views = Blueprint('views', __name__)

@views.context_processor
def inject_login_form():
    return {'form': LoginForm()}

# Home
@views.route("/")
def home():
    return render_template("index.html")

# Register
@views.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        flash("You are already registered!", "info")
        return redirect(url_for('views.home'))
    
    form = RegisterForm()
    if form.validate_on_submit():
        user = add_user(form.email.data, form.username.data, form.password.data)
        flash('Registration successful! Please log in...', 'success')
        return redirect(url_for('views.login'))
    return render_template("register.html", form=form)

# Login
@views.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        flash("You are already logged in!", "info")
        return redirect(url_for('views.home'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = verify(form.email.data, form.password.data)
        if user:
            login_user(user)
            session["current_user"] = user.username
            flash("Login successful!", "success")
            return redirect(url_for('views.home'))
        else:
            flash("Invalid email or password", "danger")
    return render_template("login.html", form=form)

# Logout
@views.route("/logout")
@login_required
def logout():
    logout_user()
    session.pop('current_user', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('views.home'))

