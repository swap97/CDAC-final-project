import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from mtcnn import MTCNN
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'best_deepfake_model.h5'
SEQ_LENGTH = 20
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = tf.keras.models.load_model(MODEL_PATH)
detector = MTCNN()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // SEQ_LENGTH)
    
    frames = []
    curr_frame = 0
    while cap.isOpened() and len(frames) < SEQ_LENGTH:
        ret, frame = cap.read()
        if not ret: break
        if curr_frame % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(frame_rgb)
            if results:
                x, y, w, h = max(results, key=lambda x: x['confidence'])['box']
                face = frame[max(0,y):y+h, max(0,x):x+w]
                face = cv2.resize(face, (128, 128))
                face = tf.keras.applications.resnet50.preprocess_input(face)
                frames.append(face)
        curr_frame += 1
    cap.release()

    while len(frames) < SEQ_LENGTH: 
        frames.append(np.zeros((128, 128, 3)))
    
    input_tensor = np.expand_dims(np.array(frames), axis=0)
    prediction = model.predict(input_tensor)
    class_idx = np.argmax(prediction)
    confidence = float(prediction[0][class_idx])
    
    label = "FAKE" if class_idx == 1 else "REAL"
    return label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
    
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        label, confidence = process_video(filepath)
        os.remove(filepath)
        
        return jsonify({
            'label': label,
            'confidence': confidence,
            'status': 'success'
        })
    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860)
