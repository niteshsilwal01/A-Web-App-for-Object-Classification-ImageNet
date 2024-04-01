from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from VGGNet16 import VGGNet16Model
from VGGNet19 import VGGNet19Model
from ResNet50 import ResNet50Model

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(file_path, model_type):
    if model_type == 'vgg16':
        model = VGGNet16Model()
    elif model_type == 'vgg19':
        model = VGGNet19Model()
    elif model_type == 'resnet50':
        model = ResNet50Model()
    else:
        return "Invalid model type", 0

    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = imagenet_utils.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    actual_prediction = imagenet_utils.decode_predictions(prediction)
    predicted_image = actual_prediction[0][0][1]
    accuracy = actual_prediction[0][0][2]
    return predicted_image, accuracy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        model_type = request.form.get('model')
        predicted_image, accuracy = predict_image(file_path, model_type)
        image_url = url_for('uploaded_file', filename=filename)
        #generate a url for index.html to render it, when user wants another prediction
        index_url = url_for('index')
        return f'''
        <h1>Prediction Model : {model_type.upper()}</h1>
        <h1>Predicted Object : {predicted_image.upper()} </h1>
        <h1>Accuracy : {accuracy*100:.2f}% </h1>
        <img src="{image_url}" alt="Uploaded Image" height="224" width="224">
        <br><br><br>

        <a href="{index_url}" class="btn btn-primary">Make Another Prediction</a>
        '''
    else:
        error = '404'
        return f''' 
        <h1>{error.upper()}</h1> <br>
        Unable to read the file. Please check file extension. The supported files are 
        <b>jpg </b>, <b>jpeg </b> and <b>png </b> only.'''

#Display Uploaded Image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
