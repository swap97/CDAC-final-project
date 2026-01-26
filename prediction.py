import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

def predict_video(video_path, model_path='best_deepfake_model.h5', seq_length=20):
    model = tf.keras.models.load_model(model_path)
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // seq_length)
    
    frames = []
    curr_frame = 0
    while cap.isOpened() and len(frames) < seq_length:
        ret, frame = cap.read()
        if not ret: break
        if curr_frame % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(frame_rgb)
            if results:
                x, y, w, h = max(results, key=lambda x: x['confidence'])['box']
                face = cv2.resize(frame[max(0,y):y+h, max(0,x):x+w], (128, 128))
                face = tf.keras.applications.resnet50.preprocess_input(face)
                frames.append(face)
        curr_frame += 1
    cap.release()

    while len(frames) < seq_length: frames.append(np.zeros((128, 128, 3)))
    
    input_tensor = np.expand_dims(np.array(frames), axis=0)
    prediction = model.predict(input_tensor)
    class_idx = np.argmax(prediction)
    
    label = "FAKE" if class_idx == 1 else "REAL"
    return label, prediction[0][class_idx]

if __name__ == "__main__":
    label, confidence = predict_video('path_to_video.mp4')
    print(f"Prediction: {label} ({confidence*100:.2f}%)")