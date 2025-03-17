from flask import Flask, render_template, request, redirect, url_for, Blueprint
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
import gdown
import h5py
import json

app = Flask(__name__)

predictions = Blueprint('predictions', __name__)

# Define the index route
@predictions.route('/')
def index():
    return render_template('index.html')

def fix_layer_names(filepath):
    with h5py.File(filepath, 'r+') as f:
        # Fixing layer names in model_weights
        if 'model_weights' in f:
            for layer in list(f['model_weights'].keys()):
                if '/' in layer:
                    new_layer_name = layer.replace('/', '_')
                    if new_layer_name not in f['model_weights']:
                        f['model_weights'][new_layer_name] = f['model_weights'][layer]
                    del f['model_weights'][layer]

        # Fixing layer names in layer_names
        if 'layer_names' in f.attrs:
            layer_names = list(f.attrs['layer_names'])
            for i in range(len(layer_names)):
                if '/' in layer_names[i].decode('utf-8'):
                    layer_names[i] = layer_names[i].replace('/', '_').encode('utf-8')
            f.attrs['layer_names'] = layer_names

        # Fixing layer names in model_config
        if 'model_config' in f.attrs:
            model_config = json.loads(f.attrs['model_config'])
            for layer in model_config['config']['layers']:
                if '/' in layer['config']['name']:
                    layer['config']['name'] = layer['config']['name'].replace('/', '_')
            f.attrs['model_config'] = json.dumps(model_config)

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Correct URL of the file on Google Drive
url = 'https://drive.google.com/uc?id=1-0o6up06YxWPdqefg_dFSqC3QLc-3A68'
# Output path where you want to save the file
output = './models/inceptiondense__model.h5'

# Download the model file if it does not exist
if not os.path.exists(output):
    gdown.download(url, output, quiet=False, fuzzy=True)

# Fix layer names before loading the model
fix_layer_names(output)

# Define a custom DepthwiseConv2D class to handle the `groups` argument
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, groups=1, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# Load the model with the custom object scope
with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
    model = tf.keras.models.load_model(output)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about/about.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            
            # Preprocess the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            # Prepare two inputs
            img_inputs = [img_array, img_array]
            
            # Make prediction
            prediction = model.predict(img_inputs)
            if prediction[0] > 0.5:
                prediction_result = "😢 The patient has pancreatic tumor."
            else:
                prediction_result = "😊 The patient doesn\'t have pancreatic cancer."
            
            # Redirect to the result page
            return render_template('predictions/result.html', prediction_text=prediction_result)
    return render_template('predictions/index.html')

@app.route("/metrics")
def metrics():
    return render_template("metrics/metrics.html")

@app.route("/flowchart")
def flowchart():
    return render_template("flowchart/flowchart.html")

if __name__ == "__main__":
    app.run(debug=True)
