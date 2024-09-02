from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from werkzeug.utils import secure_filename
from keras.utils import load_img, img_to_array
from keras.models import load_model
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "./models/catdog.h5")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
CLASS_LABELS = os.getenv("CLASS_LABELS", "Cat,Dog").split(",")

# Load the model
try:
    model = load_model(MODEL_PATH)
    print("Loaded model from disk")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def load_image(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224))
        img_tensor = img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = img_tensor / 255.0
        return img_tensor
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def main():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    file = os.path.join(UPLOAD_FOLDER, "img.jpg")
    if not os.path.exists(file):
        try:
            src = os.path.join("models", "img.jpg")
            dst = file
            shutil.copy(src, dst)
            print("Image copied")
        except Exception as e:
            print(f"Error copying image: {e}")
            return jsonify({"error": "Image copy failed"}), 500

    try:
        loaded_image = load_image(file)
        prediction = model.predict(loaded_image)
        class_id = np.argmax(prediction, axis=1)
        output = CLASS_LABELS[int(class_id)]
        result = str(output)
        print(result)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    return render_template("index.html", result=result)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        filename = secure_filename("img.jpg")
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        loaded_image = load_image(file_path)
        prediction = model.predict(loaded_image)
        class_id = np.argmax(prediction, axis=1)
        output = CLASS_LABELS[int(class_id)]
        result = str(output)
        print(result)
        return result
    except Exception as e:
        print(f"Error during file processing or prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(debug=False)  # Set debug=False for production
