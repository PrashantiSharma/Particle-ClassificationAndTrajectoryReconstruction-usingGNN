import torch
from flask import Flask, request, jsonify, render_template
from model import ClassificationModel, TrajectoryModel  # Import the models

# Initialize the Flask app
app = Flask(__name__)

# Define the input, hidden, and output dimensions (adjust according to your model)
input_dim = 10  # Example, adjust based on your model
hidden_dim = 64  # Example, adjust based on your model
output_dim = 2  # Example, adjust based on your model

# Initialize the models
classification_model = ClassificationModel(input_dim, hidden_dim, output_dim)
trajectory_model = TrajectoryModel(input_dim, hidden_dim, output_dim)

classification_model = torch.load('../models/classification_model.pth')
trajectory_model = torch.load('../models/trajectory_model.pth')


# Example route to classify a particle
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()  # Get JSON data from the request
    input_data = torch.tensor(data['features'], dtype=torch.float32)  # Convert input to tensor

    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = classification_model(input_data)  # Run the model on the input data

    predicted_class = prediction.argmax(dim=1).item()  # Get the predicted class index (argmax)

    return jsonify({'predicted_class': predicted_class})


# Example route for predicting particle trajectory
@app.route('/predict_trajectory', methods=['POST'])
def predict_trajectory():
    data = request.get_json()  # Get JSON data from the request
    input_data = torch.tensor(data['features'], dtype=torch.float32)  # Convert input to tensor

    with torch.no_grad():  # Disable gradient calculation for inference
        trajectory = trajectory_model(input_data)  # Run the model on the input data

    return jsonify({'predicted_trajectory': trajectory.tolist()})


# Route to check if the server is running
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    # Process the file with the models (you can replace this with actual file processing)
    classification_result = "Electron"  # Replace with actual model prediction
    trajectory_result = "x: 0, y: 1, z: 3"  # Replace with actual trajectory prediction

    return render_template('index.html', classification_result=classification_result, trajectory_result=trajectory_result)
if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
