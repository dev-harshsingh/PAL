from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import pickle
import os
from vark_learning_style_predictor_new import predict_learning_style

app = Flask(__name__)
CORS(app)  # Allow all origins
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vark.db'))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    college_id = db.Column(db.String(50), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer)
    course = db.Column(db.String(50))
    topic = db.Column(db.String(50))
    difficulty_level = db.Column(db.String(20))
    video_time = db.Column(db.Integer)
    article_time = db.Column(db.Integer)
    hands_on_time = db.Column(db.Integer)
    quiz_time = db.Column(db.Integer)
    quiz_score = db.Column(db.Integer)
    progress = db.Column(db.Integer)
    attempts = db.Column(db.Integer)
    learning_style = db.Column(db.String(20))

try:
    with open('model_accuracies.pkl', 'rb') as f:
        model_accuracies = pickle.load(f)
except FileNotFoundError:
    print("Error: model_accuracies.pkl not found. Run vark_learning_style_predictor_new.py first.")
    exit(1)

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to VARK Learning Style Predictor', 
                    'options': ['Remote User', 'Service Provider']})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    print("Login request:", data)
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')
    
    user = User.query.filter_by(email=email, role=role).first()
    if user and check_password_hash(user.password_hash, password):
        return jsonify({'message': 'Login successful', 'user_id': user.id, 'role': user.role})
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    print("Register request:", data)
    name = data.get('name')
    email = data.get('email')
    mobile = data.get('mobile')
    college_id = data.get('college_id')
    password = data.get('password')
    role = data.get('role')
    
    if not all([name, email, mobile, college_id, password, role]):
        return jsonify({'message': 'Missing required fields'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already registered'}), 400
    
    try:
        password_hash = generate_password_hash(password)
        user = User(name=name, email=email, mobile=mobile, college_id=college_id, 
                    password_hash=password_hash, role=role)
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'Registration successful', 'user_id': user.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Registration failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Predict request:", data)
    user_id = data.get('user_id')
    user = User.query.get(user_id)
    if not user or user.role != 'user':
        return jsonify({'message': 'Unauthorized'}), 403
    
    try:
        learning_style = predict_learning_style(
            age=data['age'],
            course=data['course'],
            topic=data['topic'],
            difficulty_level=data['difficulty_level'],
            video_time=data['video_time'],
            article_time=data['article_time'],
            hands_on_time=data['hands_on_time'],
            quiz_time=data['quiz_time'],
            quiz_score=data['quiz_score'],
            progress=data['progress'],
            attempts=data['attempts']
        )
        
        prediction = Prediction(
            user_id=user_id,
            age=data['age'],
            course=data['course'],
            topic=data['topic'],
            difficulty_level=data['difficulty_level'],
            video_time=data['video_time'],
            article_time=data['article_time'],
            hands_on_time=data['hands_on_time'],
            quiz_time=data['quiz_time'],
            quiz_score=data['quiz_score'],
            progress=data['progress'],
            attempts=data['attempts'],
            learning_style=learning_style
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({'learning_style': learning_style})
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 400

@app.route('/accuracies', methods=['GET'])
def accuracies():
    return jsonify(model_accuracies)

@app.route('/predictions', methods=['GET'])
def predictions():
    user_id = request.args.get('user_id')
    role = request.args.get('role')
    
    if role == 'admin':
        predictions = Prediction.query.all()
    else:
        predictions = Prediction.query.filter_by(user_id=user_id).all()
    
    result = [{
        'id': p.id,
        'user_id': p.user_id,
        'age': p.age,
        'course': p.course,
        'topic': p.topic,
        'difficulty_level': p.difficulty_level,
        'video_time': p.video_time,
        'article_time': p.article_time,
        'hands_on_time': p.hands_on_time,
        'quiz_time': p.quiz_time,
        'quiz_score': p.quiz_score,
        'progress': p.progress,
        'attempts': p.attempts,
        'learning_style': p.learning_style
    } for p in predictions]
    
    return jsonify(result)

if __name__ == '__main__':
    try:
        with app.app_context():
            db.drop_all()
            db.create_all()
            print(f"Database created at {db_path}: tables user, prediction")
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        exit(1)
    app.run(debug=True)