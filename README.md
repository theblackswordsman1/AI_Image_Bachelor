# Database
set FLASK_APP=app.py
flask db init
flask db migrate -m "Initial migration"
flask db upgrade

# Delete old db if recreating
del instance\app.db

# Run app (and start env)
.\env\Scripts\Activate
python app.py