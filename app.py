import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
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

# Load model and detector once at startup
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
detector = MTCNN()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    """Extracted logic from your prediction.py"""
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
                # Get the face with highest confidence
                x, y, w, h = max(results, key=lambda x: x['confidence'])['box']
                # Ensure coordinates are within bounds
                face = frame[max(0,y):y+h, max(0,x):x+w]
                face = cv2.resize(face, (128, 128))
                face = tf.keras.applications.resnet50.preprocess_input(face)
                frames.append(face)
        curr_frame += 1
    cap.release()

    # Padding if video is too short
    while len(frames) < SEQ_LENGTH: 
        frames.append(np.zeros((128, 128, 3)))
    
    input_tensor = np.expand_dims(np.array(frames), axis=0)
    prediction = model.predict(input_tensor)
    class_idx = np.argmax(prediction)
    confidence = float(prediction[0][class_idx])
    
    label = "FAKE" if class_idx == 1 else "REAL"
    return label, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            label, confidence = process_video(filepath)
            # Clean up file after processing
            os.remove(filepath)
            
            return jsonify({
                'prediction': label,
                'confidence': f"{confidence * 100:.2f}%",
                'status': 'success'
            })
        except Exception as e:
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == "__main__":
    # Use threaded=False if MTCNN/TF causes memory conflicts on some systems
    app.run(host='0.0.0.0', port=5000, debug=False)