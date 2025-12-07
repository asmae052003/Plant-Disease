# Plant Disease Classification

This project is a deep learning-based application for detecting plant diseases from images. It uses a hybrid model combining Vision Transformer (ViT) for global features and EfficientNet for local patch features, fused with an attention mechanism.

## Features

- **Advanced Hybrid Model**: Combines ViT (Global) and EfficientNet (Local) for robust feature extraction.
- **Attention Mechanisms**: Uses patch attention and fusion attention to weigh important regions and features.
- **Web Interface**: User-friendly Flask web app for easy image uploading and prediction.
- **Support**: Detects 38 different classes of plant diseases and healthy states (Apple, Corn, Grape, Tomato, etc.).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. Install the required packages:
    ```bash
    pip install torch torchvision flask pillow numpy
    ```
    *(Note: It's recommended to use a virtual environment)*

3.  **Model Checkpoint:**
    Place your trained model file `best_model_final.pth` in the project root directory.

## Usage

1.  **Run the Flask application:**
    ```bash
    python app.py
    ```

2.  **Access the interface:**
    Open your web browser and go to `http://127.0.0.1:5000/`.

3.  **Predict:**
    Upload an image of a plant leaf to get a disease prediction and confidence score.

## Project Structure

- `app.py`: Main Flask application file.
- `model.py`: Definition of the `AdvancedLocalGlobalNet` model and configuration.
- `inspect_model.py`: Script to inspect model architecture (and definition in this version).
- `templates/`: HTML templates for the web interface.
- `static/`: Static files (CSS, uploads).
- `notebooks d'entrainement/`: Jupyter notebooks used for training the model.

## Model Details

The model architecture is defined in `model.py` (and `inspect_model.py`) and features:
- **Global Branch**: ViT-B/16 pretrained on ImageNet.
- **Local Branch**: EfficientNet-B0 extracting features from high-resolution patches.
- **Fusion**: Attention-based fusion of global and local feature vectors.

## License

[Specify License Here]
