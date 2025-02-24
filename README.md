# Database
set FLASK_APP=app.py
flask db init
flask db migrate -m "Initial migration"
flask db upgrade