# Env (with python version supported by TensorFlow)

py -3.10 -m venv env
.\env\Scripts\Activate

# Requirements

pip freeze > requirements.txt
pip install -r requirements.txt

# Database

set FLASK_APP=app.py
flask db init
flask db migrate -m "Initial migration"
flask db upgrade

# Delete old db if recreating

del instance\app.db

# Run app

python app.py
