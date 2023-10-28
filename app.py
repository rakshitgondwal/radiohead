from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
import video_to_image
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

save_dir = "save"

IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
SEED = 12

# Declare test_generator as a global variable to make it accessible within functions.
test_generator = None

###################### LOADING MODEL ############################################
def load_video_analysis_model():
    model_architecture_dirname = os.path.dirname(__file__)
    model_architecture_filename = os.path.join(model_architecture_dirname, "model\CNN_model.json")
    model_weights_dirname = os.path.dirname(__file__)
    model_weights_filename = os.path.join(model_weights_dirname, "model\CNN_model_weights.h5")
    with open(model_architecture_filename, 'r') as json_file:
        loaded_model_json = json_file.read()
    video_analysis_model = model_from_json(loaded_model_json)
    video_analysis_model.load_weights(model_weights_filename)
    return video_analysis_model

######################## PREPROCESSING ########################################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

def preprocessing():
    preprocess_fun = tf.keras.applications.densenet.preprocess_input

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                preprocessing_function=preprocess_fun
                                )
    print("below test_datagen")
    global test_generator  # Use the global variable for test_generator
    test_generator = test_datagen.flow_from_directory(directory=video_to_image.test_dir,
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=False,
                                                      seed= SEED,
                                                      color_mode="rgb",
                                                      class_mode="categorical",
                                                      )
    print("end of preprocessing")

########################################### PREDICTION ##############################################

def prediction(frame_directory, model):
    preprocessing()
    y_preds = model.predict(test_generator)
    y_test = test_generator.classes
    print(y_preds)
    print(y_test)

########################################### ROUTES #################################################

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        file = request.files['video']
        if file:
            filename = secure_filename(file.filename)

            if not allowed_file(filename):
                return jsonify({"error": "Invalid file format. Please upload a valid video file."})
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            frame_directory = video_to_image.save_frame(file_path, save_dir)
            video_analysis_model = load_video_analysis_model()

            try:
                label = prediction(frame_directory, video_analysis_model)
                return jsonify({"label": label})
            except Exception as e:
                print(f"Invalid file submitted {filename}")

    return jsonify({"error": "Invalid request"})

if __name__ == '__main__':
    app.run(debug=True)
