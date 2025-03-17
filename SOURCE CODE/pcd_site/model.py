import tensorflow as tf
import numpy as np
import gdown
import os
import h5py
import json
import cv2

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

def fix_model_config(filepath):
    with h5py.File(filepath, 'r+') as f:
        if 'model_config' in f.attrs:
            model_config = json.loads(f.attrs['model_config'])
            for layer in model_config['config']['layers']:
                if 'config' in layer and 'groups' in layer['config']:
                    del layer['config']['groups']
            f.attrs['model_config'] = json.dumps(model_config)

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Correct URL of the file on Google Drive
url = 'https://drive.google.com/uc?id=1-0o6up06YxWPdqefg_dFSqC3QLc-3A68'
# Output path where you want to save the file
output = './models/inceptiondense__model.h5'

# Download the model file
gdown.download(url, output, quiet=False, fuzzy=True)

# Check if the file is correctly downloaded by checking its size
if os.path.getsize(output) < 10000:  # Adjust size as per your actual file size
    raise ValueError("Downloaded file size is too small, indicating an error in downloading.")

# Fix layer names before loading the model
fix_layer_names(output)

# Fix model configuration before loading the model
fix_model_config(output)

# Load the saved model
model = tf.keras.models.load_model(output)

def predict_cancer(preprocessed_image):
    # Reshape preprocessed image to match model input shape
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    prediction = model.predict([preprocessed_image, preprocessed_image])
    result = '😢 The patient has pancreatic tumor.' if prediction[0] > 0.5 else '😊 The patient doesn\'t have pancreatic cancer.'
    return result

def preprocess_input(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess_input(img)
    img = np.stack((img,)*3, axis=-1)  # Convert grayscale to RGB by stacking
    return img
