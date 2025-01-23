# Particle-ClassificationAndTrajectoryReconstruction-usingGNN


# Particle Classification and Trajectory Prediction using GNNs

This project involves building a web application to classify particles and predict their trajectories using machine learning models. The application uses Flask as a backend to handle requests and serve an interface for users to upload data for classification and trajectory prediction. The models are built with PyTorch and use Graph Neural Networks (GNNs) for particle classification and trajectory reconstruction.

## Features
- **Particle Classification**: Classify particles based on given input using a trained machine learning model.
- **Trajectory Prediction**: Predict the trajectory of the particles using another machine learning model.
- **Web Interface**: Users can upload particle data and view the results of classification and trajectory prediction.

## Technologies Used
- **Backend**: Flask
- **Frontend**: HTML, CSS (Responsive Web Design)
- **Modeling**: PyTorch (for machine learning models)
- **Deployment**: Local (Development)

## Project Structure


ParticleClassificationAndRetracking/
├── app/
│   ├── app.py                     # Flask app
│   ├── templates/
│   │   └── index.html             # HTML template for front-end
│   └── static/
│       └── styles.css             # CSS file for styling
├── models/
│   └── classification_model.pth   # Pre-trained classification model
│   └── trajectory_model.pth       # Pre-trained trajectory prediction model
└── requirements.txt               # Python dependencies


## Requirements

- Python 3.6 or above
- PyTorch
- Flask

### Install Dependencies

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/ParticleClassificationAndRetracking.git
    cd ParticleClassificationAndRetracking
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv .env
    source .env/bin/activate  # On Windows use .env\Scripts\activate
    ```

3. Install the required Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

4. Ensure that the models (`classification_model.pth` and `trajectory_model.pth`) are placed in the `models/` folder.

## Running the Application

1. Navigate to the app folder and run the Flask app:

    ```bash
    cd app
    python app.py
    ```

2. The application will be available at `http://127.0.0.1:5000/`.

3. Open this URL in your browser to access the front-end. Upload particle data, and you will get the predicted classification and trajectory.

## API Routes

### `/`
- **Method**: `GET`
- **Description**: Renders the main index page where the user can upload particle data for classification and trajectory prediction.

### `/upload`
- **Method**: `POST`
- **Description**: Accepts file upload, processes it using the classification and trajectory prediction models, and returns the results to the user.

## License

This project is licensed under the MIT License.

## Acknowledgments

- PyTorch for machine learning models.
- Flask for building the web application.
- TrackML Particle tracking challenge dataset (used for model training).

---

Feel free to customize it with any extra details specific to your project!