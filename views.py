from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify, current_app
import flask
from database import list_users, verify, delete_user_from_db, add_user
from flask_login import login_user, logout_user, login_required, current_user, login_manager
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from extensions import db
from models import User, RegisterForm, LoginForm
import os
# from utils.code_generator import CNNCodeGenerator
import functools
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import tensorflow_datasets as tfds
import shutil
from PIL import Image
import io
import numpy as np
import threading
from queue import Queue
import time


views = Blueprint('views', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def get_current_step():
    return session.get('current_step', 1)

training_status = {
    'is_running': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'accuracy': None,
    'model_path': None,
    'error': None,
    'output_queue': Queue()
}

@views.context_processor
def inject_template_context():
    uploaded_classes = session.get('uploaded_classes', [])
    done_steps = session.get('done_steps', [])

    if 1 in done_steps and len(uploaded_classes) < 2:
        done_steps.remove(1)
        session['done_steps'] = done_steps

    return {
        'current_step': get_current_step(),
        'done_steps': done_steps,
        'uploaded_classes': uploaded_classes,
        'request': request,
        'is_authenticated': current_user.is_authenticated if hasattr(current_user, 'is_authenticated') else False
    }

# Home
@views.route("/")
def home():
    session['current_step'] = 1
    session['done_steps'] = []
    session['uploaded_classes'] = []
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
def logout():
    if current_user.is_authenticated:
        logout_user()
        session.pop('current_user', None)
        flash('You have been logged out!', 'info')
    return redirect(url_for('views.home'))

# Cnn image classification
@views.route("/cnn-classification", methods=["GET", "POST"])
def cnn_classification():
    if 'current_step' not in session:
        session['current_step'] = 1
    return render_template("cnn_classification.html")

# Step 1, data upload
@views.route("/upload", methods=["GET", "POST"])
def upload():
    session['current_step'] = 1
    uploaded_classes = session.get('uploaded_classes', [])
    done_steps = session.get('done_steps', [])

    if len(uploaded_classes) >= 2:
        if 1 not in done_steps:
            done_steps.append(1)
            session['done_steps'] = done_steps
    else:
        if 1 in done_steps:
            done_steps.remove(1)
            session['done_steps'] = done_steps

    if request.method == "POST":
        if 'dataset' in request.form:
            selected_dataset = request.form.get('dataset')
            if selected_dataset in ['mnist', 'cifar10']:
                session['uploaded_classes'] = []
                if selected_dataset == 'mnist':
                    dataset_info = {
                        'name': 'MNIST',
                        'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                        'image_shape': (28, 28, 1)
                    }
                    session['preprocessing'] = session.get('preprocessing', {})
                    session['preprocessing']['image_size'] = 28
                    session['preprocessing']['num_classes'] = 10
                    
                    print(f"MNIST selected, dataset_info: {dataset_info}")
                    print(f"Session preprocessing: {session['preprocessing']}")
                elif selected_dataset == 'cifar10':
                    dataset_info = {
                        'name': 'CIFAR-10',
                        'classes': ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck'],
                        'image_shape': (32, 32, 3)
                    }
                    session['preprocessing'] = session.get('preprocessing', {})
                    session['preprocessing']['image_size'] = 32
                    session['preprocessing']['num_classes'] = 10
                
                session['selected_dataset'] = selected_dataset
                session['dataset_info'] = dataset_info
                
                uploaded_classes = []
                for class_name in dataset_info['classes']:
                    if selected_dataset == 'mnist':
                        image_count = 7000
                    elif selected_dataset == 'cifar10':
                        image_count = 6000

                    uploaded_classes.append({
                        'name': class_name,
                        'count': image_count,
                        'path': f"built-in/{selected_dataset}/{class_name}",
                        'is_built_in': True
                    })
                
                session['uploaded_classes'] = uploaded_classes
                
                done_steps = session.get('done_steps', [])
                if 1 not in done_steps:
                    done_steps.append(1)
                session['done_steps'] = done_steps
                session.modified = True
                
                flash(f"Successfully loaded the {dataset_info['name']} dataset with {len(dataset_info['classes'])} classes", "success")
                flask.session.modified = True
                response = redirect(url_for('views.preprocess'))
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                return response
            
        elif 'folder[]' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No folder selected'
            }), 400
        
        files = request.files.getlist('folder[]')
        if not files:
            return jsonify({
                'status': 'error',
                'message': 'Empty folder'
            }), 400

        class_name = request.form.get('class_name')
        if not class_name:
            return jsonify({
                'status': 'error',
                'message': 'Class name is required'
            }), 400
        
        uploaded_classes = session.get('uploaded_classes', [])
        if any(c['name'] == class_name for c in uploaded_classes):
            return jsonify({
                'status': 'error',
                'message': f"Class '{class_name}' already exists"
            }), 400
        
        base_dir = os.path.join(
            current_app.config['UPLOAD_FOLDER'],
            session.get('current_user', 'anonymous')
        )
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        image_count = 0
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename.split('/')[-1])
                file.save(os.path.join(class_dir, filename))
                image_count += 1
        
        if image_count > 0:
            uploaded_classes.append({
                'name': class_name,
                'count': image_count,
                'path': class_dir
            })
            session['uploaded_classes'] = uploaded_classes
            
            if len(uploaded_classes) >= 2:
                done_steps = session.get('done_steps', [])
                if 1 not in done_steps:
                    done_steps.append(1)
                    session['done_steps'] = done_steps
            
            return jsonify({
                'status': 'success',
                'message': f"Successfully uploaded {image_count} images for class '{class_name}'",
                'classes': uploaded_classes
            })
         
    return render_template("cnn_classification.html", uploaded_classes=session.get('uploaded_classes', []))

# Remove class
@views.route("/remove-class/<class_name>", methods=["POST"])
def remove_class(class_name):
    uploaded_classes = session.get('uploaded_classes', [])
    done_steps = session.get('done_steps', [])
    
    for cls in uploaded_classes:
        if cls['name'] == class_name:
            if not cls.get('is_built_in', False):
                import shutil
                try:
                    shutil.rmtree(cls['path'])
                except FileNotFoundError:
                    pass
            
            uploaded_classes.remove(cls)
            break
    
    session['uploaded_classes'] = uploaded_classes
    if len(uploaded_classes) < 2 and 1 in done_steps:
        done_steps.remove(1)
        session['done_steps'] = done_steps

    return jsonify({
        'status': 'success',
        'classes': uploaded_classes
    })

# Step 2, preprocessing
@views.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    session['current_step'] = 2
    if request.method == "POST":
        session['preprocessing'] = {
            'image_size': request.form.get('image_size'),
            'augmentation': request.form.getlist('augmentation[]'),
            'train_split': request.form.get('train_split'),
            'val_split': request.form.get('val_split'),
            'test_split': request.form.get('test_split')
        }
        done_steps = session.get('done_steps', [])

        if 2 not in done_steps:
            done_steps.append(2)
            session['done_steps'] = done_steps

        return redirect(url_for("views.model_config"))
    
    return render_template("cnn_classification.html")


# Step 3, model config
@views.route("/model-config", methods=["GET", "POST"])
def model_config():
    session['current_step'] = 3
    if request.method == "POST":
        filters = request.form.getlist('filters[]')
        kernel_sizes = request.form.getlist('kernel_size[]')
        dense_units = request.form.getlist('dense_units[]')
        
        session['model_config'] = {
            'filters': [int(f) for f in filters if f],
            'kernel_size': [int(k) for k in kernel_sizes if k],
            'dense_units': [int(d) for d in dense_units if d],
            'padding': request.form.get('padding', 'same'),
            'activation': request.form.get('activation', 'relu'),
            'batch_norm': 'batch_norm' in request.form,
            'dropout_rate': float(request.form.get('dropout_rate', 0.5)),
            'pool_size': int(request.form.get('pool_size', 2)),
            'pool_type': request.form.get('pool_type', 'max')
        }
        
        done_steps = session.get('done_steps', [])
        if 3 not in done_steps:
            done_steps.append(3)
            session['done_steps'] = done_steps

        return redirect(url_for("views.training_config"))
    
    return render_template("cnn_classification.html")

# Step 4, training config
@views.route("/training-config", methods=["GET", "POST"])
def training_config():
    session['current_step'] = 4
    if request.method == "POST":
        try:
            epochs = max(1, min(1000, int(request.form.get('epochs', 10))))
            batch_size = max(1, min(256, int(request.form.get('batch_size', 32))))
            learning_rate = max(0.0001, min(0.1, float(request.form.get('learning_rate', 0.001))))
            patience = max(1, min(50, int(request.form.get('patience', 5))))
            optimizer = 'Adam'

            session['training_config'] = {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'optimizer': optimizer,
                'patience': patience
            }

            done_steps = session.get('done_steps', [])
            if 4 not in done_steps:
                done_steps.append(4)
                session['done_steps'] = done_steps

            if 'generate' in request.form:
                return redirect(url_for('views.evaluation'))
            
            flash('Training configuration saved successfully!', 'success')
            return redirect(url_for('views.training_config'))

        except (ValueError, TypeError) as e:
            flash(f'Invalid training configuration: {str(e)}', 'error')
            return redirect(url_for('views.training_config'))
            
    return render_template("cnn_classification.html")

def generate_model_code(session_data):
    generator = CNNCodeGenerator(session_data)
    return generator.generate_complete_code()


# Model training
@views.route("/evaluation", methods=["GET"])
def evaluation():
    session['current_step'] = 5
    
    if not all(key in session for key in ['preprocessing', 'model_config', 'training_config']):
        flash('Please complete all configuration steps first', 'error')
        return redirect(url_for('views.upload'))
        
    generator = CNNCodeGenerator({
        'preprocessing': session.get('preprocessing', {}),
        'model_config': session.get('model_config', {}),
        'training_config': session.get('training_config', {})
    })
    
    generated_code = generator.generate_complete_code()
    return render_template("cnn_classification.html", 
                         current_step=5,
                         generated_code=generated_code,
                         done_steps=session.get('done_steps', []))

@views.route("/train-model", methods=["POST"])
def train_model():
    global training_status
    
    if training_status['is_running']:
        return jsonify({
            'status': 'error',
            'message': 'Training is already in progress'
        }), 400

    try:
        config = {
            'preprocessing': session.get('preprocessing', {}),
            'model_config': session.get('model_config', {}),
            'training_config': session.get('training_config', {}),
            'selected_dataset': session.get('selected_dataset'),
            'dataset_info': session.get('dataset_info'),
            'uploaded_classes': session.get('uploaded_classes', [])
        }

        training_status.update({
            'is_running': True,
            'current_epoch': 0,
            'total_epochs': config['training_config'].get('epochs', 10),
            'accuracy': None,
            'model_path': None,
            'error': None,
            'output_queue': Queue()
        })

        thread = threading.Thread(
            target=run_training,
            args=(config, training_status)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'status': 'success',
            'message': 'Training started'
        }), 200

    except Exception as e:
        training_status['is_running'] = False
        training_status['error'] = str(e)
        return jsonify({
            'status': 'error',
            'message': f'Failed to start training: {str(e)}'
        }), 500

@views.route("/training-status")
def check_training_status():
    global training_status
    
    messages = []
    while not training_status['output_queue'].empty():
        messages.append(training_status['output_queue'].get())

    if training_status['error']:
        return jsonify({
            'status': 'error',
            'message': training_status['error']
        }), 400
    
    if training_status['is_running']:
        return jsonify({
            'status': 'running',
            'current_epoch': training_status['current_epoch'],
            'total_epochs': training_status['total_epochs'],
            'message': '\n'.join(messages) if messages else 'Training in progress...'
        }), 200
    
    if training_status['accuracy'] is not None:
        return jsonify({
            'status': 'completed',
            'accuracy': training_status['accuracy'],
            'model_path': training_status['model_path'],
            'message': '\n'.join(messages) if messages else 'Training completed!'
        }), 200
    
    return jsonify({
        'status': 'idle',
        'message': 'Training not started'
    }), 200


@views.route("/update-step", methods=["POST"])
def update_step():
    try:
        data = request.json
        if 'step' in data:
            new_step = int(data['step'])

            if new_step == 2:
                if 'uploaded_classes' not in session or len(session.get('uploaded_classes', [])) < 2:
                        return jsonify({
                            'status': 'error',
                            'message': 'Please upload at least 2 classes before continuing'
                        })
                
                done_steps = session.get('done_steps', [])
                if 1 not in done_steps:
                    done_steps.append(1)
                session['done_steps'] = done_steps
                session['current_step'] = 2

                return jsonify({'status': 'success'})
            
        return jsonify({'status': 'error', 'message': 'Invalid step update'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# Empty pages
# Temp classification
@views.route("/temp-classification")
def temp_classification():
    return render_template("temp_classification.html")

# Other classification
@views.route("/other-classification")
def other_classification():
    return render_template("other_classification.html")

# My projects
@views.route("/my-projects")
def my_projects():
    return render_template("my_projects.html")