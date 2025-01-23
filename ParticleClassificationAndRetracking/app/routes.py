from flask import Blueprint, jsonify

api = Blueprint('api', __name__)

@api.route('/')
def home():
    return jsonify({'message': 'Welcome to My Flask App!'})

@api.route('/predict', methods=['POST'])
def predict():
    # Placeholder for ML prediction
    return jsonify({'result': 'Prediction result here'})
