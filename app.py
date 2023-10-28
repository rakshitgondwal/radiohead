from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
import cv2
import os

app = Flask(__name__)

def load_video_analysis_model():
    model_architecture_dirname = os.path.dirname(__file__)
    model_architecture_filename = os.path.join(model_architecture_dirname, "model\CNN_model.json")
    model_weights_dirname = os.path.dirname(__file__)
    model_weights_filename = os.path.join(model_weights_dirname, "model\CNN_model_weights.h5")
    print(model_architecture_filename)
    print(model_weights_filename)
    with open(model_architecture_filename, 'r') as json_file:
        loaded_model_json = json_file.read()
    video_analysis_model = model_from_json(loaded_model_json)
    video_analysis_model.load_weights(model_weights_filename)
    
    return video_analysis_model

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'GET':
        load_video_analysis_model()
        return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True)