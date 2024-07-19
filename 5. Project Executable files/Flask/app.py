from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Load the model once at the beginning
model = load_model('vgg16_model.h5')

@app.route('/predict', methods=["GET", "POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/uploads')

        # Ensure the uploads directory exists
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        filepath = os.path.join(upload_path, f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(180, 180))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        pred = np.argmax(preds, axis=1)

        index = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
        result = index[pred[0]]

        return render_template('output.html', prediction=result)

    return render_template('input.html')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/details')
def details():
    return render_template('details.html')
@app.route('/images')
def images():
    return render_template('images.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

if __name__ == "__main__":
    app.run(debug=True)
